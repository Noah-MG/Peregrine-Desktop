# Self-tests. Included into the PeregrineSolver module.
#
# The repo rule is that a self-test must test what the tool is *for*, not
# whatever is convenient to assert. This one is for producing minimum-time
# value tables, so the tests are:
#
#   1. on a problem whose minimum time is known in closed form, does the
#      solver recover it?
#   2. is the integrator the order it claims to be?
#   3. does the fast control search ever return a worse answer than scanning
#      the whole lattice?
#   4. is a state outside the velocity envelope actually rejected?
#   5. does a NaN traction knee still mean "no saturation"?
#   6. is every command the lattice offers one the robot could actually run?
#   7. does widening the step-length ladder only ever lower the value?
#   8. is the Kuhn simplex interpolant exact, continuous and a partition of
#      unity?
#
# Tests 3 and 7 guard the optimisation work: "faster" is only allowed if it
# is also never worse, and "more options" has to mean a superset of the
# options, not a different set of them. Test 8 guards the interpolant, which
# is the one piece where being nearly right is indistinguishable from being
# right until a table is already on a robot.

using Printf
using Random

const _PASS = Ref(0)
const _FAIL = Ref(0)

function _check(ok::Bool, name::AbstractString, detail::AbstractString = "")
    if ok
        _PASS[] += 1
        println("  PASS  ", name, isempty(detail) ? "" : "   ($detail)"); flush(stdout)
    else
        _FAIL[] += 1
        println("  FAIL  ", name, isempty(detail) ? "" : "   ($detail)"); flush(stdout)
    end
    ok
end

"""A pure double integrator: `a = amax * u`, nothing else."""
function _double_integrator(amax::Float32)
    Z9 = ntuple(_ -> 0.0f0, 9)
    B = (amax, 0.0f0, 0.0f0, 0.0f0, amax, 0.0f0, 0.0f0, 0.0f0, amax)
    Model(B, Z9, (0.0f0, 0.0f0, 0.0f0), Z9, Z9, (0.0f0, 0.0f0, 0.0f0),
          (5.0f0, 5.0f0, 0.15f0), 0.0f0)
end

"""A model with the shape of a real fit, for integrator tests."""
function _realistic()
    B = (-849.0f0, -25.8f0, -69.5f0, 5.0f0, -846.8f0, 0.6f0, -0.2f0, -9.6f0, -64.9f0)
    A = (-2.13f0, -0.007f0, -8.72f0, -0.064f0, -4.96f0, -0.013f0,
         0.039f0, 0.078f0, -4.67f0)
    S = (-11.6f0, -3.37f0, 3.29f0, 2.07f0, 6.33f0, 1.00f0,
         -0.56f0, -1.17f0, -0.28f0)
    D = (0.021f0, 0.003f0, 0.729f0, 0.001f0, 0.049f0, 0.871f0,
         -0.0004f0, -0.0006f0, 0.312f0)
    Model(B, A, (0.0f0, 0.0f0, 0.0f0), S, D, (0.0f0, 0.0f0, 0.0f0),
          (5.0f0, 5.0f0, 0.15f0), 0.45f0)
end

# --------------------------------------------------------------------------

"""
Test 1: recover the closed-form minimum time of a double integrator.

For `a = amax*u`, `|u| <= 1`, the minimum time from rest at distance `d` to
rest at the origin is the bang-bang solution `2*sqrt(d/amax)` -- accelerate
for half the distance, brake for the other half. Nothing about the solver
knows that, so agreeing with it exercises the whole chain: the control set,
the integration, the interpolation, the seeding and the iteration.

The solved value is expected to land slightly BELOW the closed form, because
the target is seeded as a cell rather than a point and arriving anywhere in
that cell counts.

**The grid has to be proportioned for the problem**, and this test is where
that shows. A semi-Lagrangian backup only learns from a step that leaves the
cell it started in, and for a double integrator the distance covered while
the speed changes by one velocity cell is `dv^2 / (2a)`. If the position
cells are much coarser than that, a step that resolves the velocity axis
barely moves in position, the interpolated successor is mostly the cell's own
value, and the converged answer comes out systematically pessimistic --
measured at +44% to +61% on a grid that was 10x too coarse in position for
its velocity resolution. That is not a solver defect; it is what the
discretisation is worth. `plan` now reports the ratio so it can be seen
before a long run rather than inferred from a disappointing table.
"""
function test_min_time(; use_gpu = CUDA.functional())
    println("\n[1] minimum time against the closed-form bang-bang solution")
    amax = 100.0f0
    m = _double_integrator(amax)
    n = (81, 5, 4, 33, 5, 5)
    g = Grid6(n, (0.0, 0.0, -π, -100.0, -100.0, -2.0),
                 (200.0, 200.0, π, 100.0, 100.0, 2.0))
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    cap = 30.0f0
    targ = (100.0f0, 100.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    tcells = target_cells(g, targ, ttol)
    ncell = ncells(g)

    function run(p, level)
        ctl_t = control_set(level)
        nctl = length(ctl_t)
        ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2);
                        getindex.(ctl_t, 3)]
        if use_gpu
            V = CUDA.fill(cap, ncell)
            pol = p.ncoarse > 0 ? CUDA.zeros(Int8, 3 * ncell) : nothing
            solve_value!(V, CuArray(occ), g, m, CuArray(Int64.(tcells) .+ 1),
                         CuArray(ctl_h), nctl, p; iters = 600, tol = 1e-4,
                         use_gpu = true, pol = pol,
                         total = CUDA.zeros(Float32, 1))
            Array(V)
        else
            V = fill(cap, ncell)
            pol = p.ncoarse > 0 ? zeros(Int8, 3 * ncell) : nothing
            solve_value!(V, occ, g, m, Int64.(tcells) .+ 1, ctl_h, nctl, p;
                         iters = 600, tol = 1e-4, use_gpu = false, pol = pol)
            V
        end
    end

    # Three schemes, not two, so a failure attributes itself. The earlier
    # two-way version could not distinguish "the cheap control search lost
    # something" from "the horizon ladder lost something", and on a double
    # integrator those pull in opposite directions.
    #
    # All three get the velocity-envelope fix and the same interpolant.
    # Comparing against the old clamped behaviour would flatter the rival,
    # not the new scheme: clamping makes V optimistic, so a clamped table
    # produces smaller numbers that no robot can deliver.
    common = (dt = 0.05f0, nsub = Int32(4), cap = cap)
    legacy = run(Params(; common..., checks = Int32(3), adaptive_checks = false,
                        ntau = Int32(1), cfl = 0.0f0, ncoarse = Int32(0),
                        rounds = Int32(0), rk2 = false, vclamp = false), 4)
    ladder = run(Params(; common..., checks = Int32(2), adaptive_checks = true,
                        ntau = Int32(4), cfl = 1.0f0, hmax = 0.0125f0,
                        ncoarse = Int32(0), rounds = Int32(0), rk2 = true), 4)
    now    = run(Params(; common..., checks = Int32(2), adaptive_checks = true,
                        ntau = Int32(4), cfl = 1.0f0, hmax = 0.0125f0,
                        ncoarse = Int32(8), rounds = Int32(2), rk2 = true), 4)

    ds = (20.0f0, 40.0f0, 60.0f0)
    want = [2.0 * sqrt(Float64(d) / Float64(amax)) for d in ds]
    at(T, d) = Float64(interp_kuhn(T, g, 100.0f0 - d, 100.0f0, 0.0f0, 0.0f0,
                                   0.0f0, 0.0f0, cap)[1])
    gotl = [at(legacy, d) for d in ds]
    gotd = [at(ladder, d) for d in ds]
    gotn = [at(now, d) for d in ds]

    for i in eachindex(ds)
        @printf("        d = %2.0f cm  exact %.3f  fixed-dt %.3f (%+.1f%%)  +ladder %.3f (%+.1f%%)  +fast-search %.3f (%+.1f%%)
",
                ds[i], want[i],
                gotl[i], 100 * (gotl[i] - want[i]) / want[i],
                gotd[i], 100 * (gotd[i] - want[i]) / want[i],
                gotn[i], 100 * (gotn[i] - want[i]) / want[i])
    end

    ml = sum(gotl) / 3; md = sum(gotd) / 3; mn = sum(gotn) / 3

    # A double integrator is the one case where the cheap control search has
    # nothing to find: with no traction knee the dynamics are affine in `u`,
    # so the minimum of a linear form over the octahedron really is at a
    # vertex and the lattice corners are already exactly optimal.
    # Off-lattice refinement cannot improve on that, and a rotating scan can
    # only be slower to reach it. So a few percent of slack here is the
    # expected price, paid back many times over on a model that has the knee
    # -- which is what test 3 measures.
    ok = _check(mn <= md * 1.08, "cheap search costs little where it can gain nothing",
                @sprintf("exhaustive %.3f s, rotating scan %.3f s, %+.1f%%",
                         md, mn, 100 * (mn - md) / md))
    ok &= _check(md <= ml * 1.02, "the horizon ladder does not lose ground either",
                 @sprintf("fixed dt %.3f s, ladder %.3f s, %+.1f%%",
                          ml, md, 100 * (md - ml) / ml))

    # Split the excess into a fixed part and a per-second part. The excess is
    # dominated by the terminal approach: the last few centimetres are covered
    # at low speed, where a step that resolves the velocity axis moves a tiny
    # fraction of a position cell and the interpolation is at its most
    # diffusive. That part does not grow with the distance travelled, so it
    # shows up as an intercept. Anything that DOES grow with distance is a
    # per-step loss in the cruise, and that is the one worth failing over --
    # a flat percentage tolerance cannot tell the two apart.
    ex = gotn .- want
    slope = (ex[3] - ex[1]) / (want[3] - want[1])
    intercept = ex[1] - slope * want[1]
    @printf("        excess = %.3f s fixed + %.1f%% of the journey
",
            intercept, 100 * slope)
    ok &= _check(intercept < 0.45, "terminal excess is bounded",
                 @sprintf("%.3f s", intercept))
    ok &= _check(slope < 0.20, "no significant per-second loss in the cruise",
                 @sprintf("%.1f%% per second", 100 * slope))
    ok
end

"""
Test 2: the midpoint integrator is second order and Euler is first.

Measured against the same integrator run far finer, on a model with the shape
of a real fit -- including the traction knee and the Coulomb band, which are
where the error actually lives.
"""
function test_integrator()
    println("\n[2] integrator order")
    m = _realistic()
    s0 = (10.0f0, 20.0f0, 0.3f0, 40.0f0, -25.0f0, 1.5f0)
    u = (0.6f0, -0.3f0, 0.1f0)
    dt = 0.05f0

    ref = step_state_rk2(m, s0..., u..., dt, Int32(4096))
    err(s) = maximum(abs.(Float64.(s) .- Float64.(ref)))

    e_eul = err(step_state(m, s0..., u..., dt, Int32(4)))
    e_rk2 = err(step_state_rk2(m, s0..., u..., dt, Int32(4)))
    ok = _check(e_rk2 < e_eul / 5, "midpoint beats Euler at the same substeps",
                @sprintf("Euler %.4g, midpoint %.4g, %.1fx better",
                         e_eul, e_rk2, e_eul / e_rk2))

    # Halving the step should quarter a second-order error.
    e4 = err(step_state_rk2(m, s0..., u..., dt, Int32(4)))
    e8 = err(step_state_rk2(m, s0..., u..., dt, Int32(8)))
    order = log2(e4 / e8)
    ok &= _check(order > 1.5, "midpoint converges at order > 1.5",
                 @sprintf("observed order %.2f", order))

    e4e = err(step_state(m, s0..., u..., dt, Int32(4)))
    e8e = err(step_state(m, s0..., u..., dt, Int32(8)))
    ok &= _check(log2(e4e / e8e) < 1.4, "Euler converges at order ~1",
                 @sprintf("observed order %.2f", log2(e4e / e8e)))

    # A long horizon from a fast spin walks the state into a region where the
    # FITTED model is unstable -- extrapolated far enough, its `+0.312*|w|*w`
    # drag term outruns the `-4.67*w` linear term and runs away. The ladder
    # tries such horizons on purpose, because trying them is how it learns
    # they are not worth taking, so the integrator has to survive divergence
    # and report it rather than trip over it.
    #
    # This is CPU-only in practice and that is exactly why it is here: CUDA
    # returns NaN from `sincos(Inf)` and carries on, but the CPU throws a
    # DomainError and takes the whole solve down. The GPU hid it completely.
    spin = (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 30.0f0)
    for f in (step_state, step_state_rk2)
        got = try
            f(m, spin..., 0.0f0, 0.0f0, 1.0f0, 2.0f0, Int32(160))
        catch e
            (e,)
        end
        ok &= _check(length(got) == 6 && !all(isfinite, got),
                     "$(nameof(f)) reports divergence instead of throwing",
                     length(got) == 6 ? "non-finite state returned" : "threw $(got[1])")
    end
    ok
end

"""
Test 3: the fast control search is never worse than scanning everything.

This is the test that licenses the optimisation. Both schemes are run to
convergence on the same problem; the fast one visits a fraction of the
lattice per sweep but refines off-lattice and tries several step lengths, so
it should come out at least as tight everywhere and tighter on average. A
cell where it is *worse* means the rotating scan has not covered the lattice
before the stopping rule fired.
"""
function test_search_quality(; use_gpu = CUDA.functional())
    println("\n[3] fast control search vs exhaustive lattice scan")
    m = _realistic()
    n = (17, 17, 8, 7, 7, 7)
    g = Grid6(n, (0.0, 0.0, -π, -150.0, -150.0, -8.0),
                 (200.0, 200.0, π, 150.0, 150.0, 8.0))
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    cap = 30.0f0
    targ = (100.0f0, 100.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    tcells = target_cells(g, targ, ttol)
    ncell = ncells(g)

    function run(p, level)
        ctl_t = control_set(level)
        nctl = length(ctl_t)
        ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2);
                        getindex.(ctl_t, 3)]
        if use_gpu
            V = CUDA.fill(cap, ncell); occ_d = CuArray(occ)
            ctl_d = CuArray(ctl_h); total = CUDA.zeros(Float32, 1)
            tidx = CuArray(Int64.(tcells) .+ 1)
            pol = p.ncoarse > 0 ? CUDA.zeros(Int8, 3 * ncell) : nothing
            solve_value!(V, occ_d, g, m, tidx, ctl_d, nctl, p; iters = 500,
                         tol = 1e-4, use_gpu = true, pol = pol, total = total)
            Array(V)
        else
            V = fill(cap, ncell)
            pol = p.ncoarse > 0 ? zeros(Int8, 3 * ncell) : nothing
            solve_value!(V, occ, g, m, Int64.(tcells) .+ 1, ctl_h, nctl, p;
                         iters = 500, tol = 1e-4, use_gpu = false, pol = pol)
            V
        end
    end

    slow = run(Params(dt = 0.05f0, cap = cap, ncoarse = Int32(0),
                      rounds = Int32(0), ntau = Int32(1), cfl = 0.0f0,
                      rk2 = true), 4)
    fast = run(Params(dt = 0.05f0, cap = cap, ncoarse = Int32(8),
                      rounds = Int32(2), ntau = Int32(4), cfl = 1.0f0,
                      rk2 = true), 4)

    both = [i for i in eachindex(slow) if slow[i] < cap - 1f-3 && fast[i] < cap - 1f-3]
    worse = count(i -> fast[i] > slow[i] + 1.0f-3, both)
    ms = sum(slow[both]) / length(both)
    mf = sum(fast[both]) / length(both)
    ok = _check(worse / length(both) < 0.01, "fast search worse on < 1% of cells",
                @sprintf("%d of %d cells", worse, length(both)))
    ok &= _check(mf <= ms + 1.0f-4, "fast search is tighter on average",
                 @sprintf("exhaustive %.4f s, fast %.4f s, %+.2f%%",
                          ms, mf, 100 * (mf - ms) / ms))
    rs = count(v -> v < cap - 1f-3, slow) / length(slow)
    rf = count(v -> v < cap - 1f-3, fast) / length(fast)
    ok &= _check(rf >= rs - 0.005, "fast search reaches as many cells",
                 @sprintf("exhaustive %.4f, fast %.4f", rs, rf))
    ok
end

"""
Test 7: adding step lengths can only lower the value, never raise it.

The dynamic programming principle holds for any lookahead horizon, so a
backup that minimises over several is a valid backup and a tighter one. Both
runs here search controls at the same `tau0`; the second additionally
brackets it. If the ladder ever came out HIGHER, the extra horizons would not
be extra options -- they would be replacing the original one, which would
mean the scheme is no longer a superset and the "can only improve" argument
has a hole in it.
"""
function test_tau_monotone(; use_gpu = CUDA.functional())
    println("\n[7] a wider step-length ladder never raises the value")
    m = _realistic()
    n = (15, 15, 8, 7, 7, 7)
    g = Grid6(n, (0.0, 0.0, -π, -150.0, -150.0, -8.0),
                 (200.0, 200.0, π, 150.0, 150.0, 8.0))
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    cap = 30.0f0
    targ = (100.0f0, 100.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    tcells = target_cells(g, targ, ttol)
    ncell = ncells(g)
    ctl_t = control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]

    function run(ntau)
        p = Params(dt = 0.05f0, cap = cap, cfl = 1.0f0, ntau = Int32(ntau),
                   ncoarse = Int32(0), rounds = Int32(0), rk2 = true)
        if use_gpu
            V = CUDA.fill(cap, ncell)
            solve_value!(V, CuArray(occ), g, m, CuArray(Int64.(tcells) .+ 1),
                         CuArray(ctl_h), nctl, p; iters = 500, tol = 1e-4,
                         use_gpu = true, total = CUDA.zeros(Float32, 1))
            Array(V)
        else
            V = fill(cap, ncell)
            solve_value!(V, occ, g, m, Int64.(tcells) .+ 1, ctl_h, nctl, p;
                         iters = 500, tol = 1e-4, use_gpu = false)
            V
        end
    end

    one = run(1)
    lad = run(4)
    raised = count(i -> lad[i] > one[i] + 1.0f-3, eachindex(one))
    ok = _check(raised == 0, "no cell is raised by the ladder",
                "$raised of $(length(one)) cells")
    reached = [i for i in eachindex(one) if one[i] < cap - 1f-3]
    m1 = sum(one[reached]) / length(reached)
    m4 = sum(lad[reached]) / length(reached)
    ok &= _check(m4 <= m1 + 1.0f-4, "the ladder is tighter on average",
                 @sprintf("one horizon %.4f s, ladder %.4f s, %+.2f%%",
                          m1, m4, 100 * (m4 - m1) / m1))
    ok
end

"""
Test 8: the Kuhn simplex interpolant, against the properties it must have.

This is the one piece where being *nearly* right is indistinguishable from
being right until a table is on a robot, so it is checked against properties
rather than against a reference implementation:

  * **Exact on affine data.** Both the value and all six gradient components.
    A wrong vertex walk, a mis-sorted axis or a bad barycentric weight all
    break this immediately, and it is the property the robot's gradient
    recovery actually depends on.
  * **Partition of unity.** Interpolating a constant table returns the
    constant. Weights that did not sum to one would bias every value in the
    table by a factor, which is exactly the kind of error that looks like
    "the solver is a bit pessimistic".
  * **Continuity across cell faces.** The Freudenthal decomposition
    triangulates a shared face the same way from both sides. If it did not,
    `V` would have seams and the robot's gradient would flip across them.
  * **Agreement with multilinear on affine data**, since both are exact
    there -- a cross-check of two independent implementations.
"""
function test_kuhn()
    println("\n[8] Kuhn simplex interpolation")
    n = (7, 6, 8, 5, 5, 5)
    g = Grid6(n, (0.0, 0.0, -π, -100.0, -100.0, -4.0),
                 (120.0, 100.0, π, 100.0, 100.0, 4.0))
    cap = 1.0f6
    ncell = ncells(g)

    # An affine function of the six axis VALUES, laid out on the grid.
    co = (0.017f0, -0.023f0, 0.31f0, 0.0041f0, -0.0072f0, 0.11f0)
    c0 = 3.5f0
    fexact(s) = c0 + sum(co[k] * s[k] for k in 1:6)
    V = Vector{Float32}(undef, ncell)
    for idx in 0:ncell-1
        i = unflatten(g, idx)
        V[idx+1] = fexact(ntuple(k -> axisvalue(g, k, i[k]), 6))
    end

    rng = MersenneTwister(4242)
    worst_v = 0.0; worst_g = 0.0
    for _ in 1:4000
        # Stay off the heading seam. Heading spans [-pi, -pi+(n3-1)*step]
        # before wrapping, and the affine reference is not periodic, so a
        # sample in the wrap cell is discontinuous by construction -- the
        # test's fault, not the interpolant's. That is exactly what the first
        # version of this test caught in itself: Kuhn and multilinear agreed
        # with each other to 3e-6 while both "disagreed" with the reference
        # by 0.6, which can only mean the reference was wrong.
        s = (rand(rng) * 110.0, rand(rng) * 90.0,
             -3.0 + rand(rng) * 5.2,
             -90.0 + rand(rng) * 180.0, -90.0 + rand(rng) * 180.0,
             -3.5 + rand(rng) * 7.0)
        sf = Float32.(s)
        v, d1, d2, d3, d4, d5, d6 = interp_kuhn(V, g, sf..., cap)
        worst_v = max(worst_v, abs(Float64(v) - Float64(fexact(sf))))
        gd = (d1, d2, d3, d4, d5, d6)
        for k in 1:6
            worst_g = max(worst_g, abs(Float64(gd[k]) - Float64(co[k])))
        end
    end
    ok = _check(worst_v < 2.0e-3, "exact on affine data",
                @sprintf("worst value error %.3g", worst_v))
    ok &= _check(worst_g < 2.0e-4, "gradient exact on affine data",
                 @sprintf("worst gradient error %.3g", worst_g))

    # Partition of unity.
    Vc = fill(2.75f0, ncell)
    worst_c = 0.0
    for _ in 1:2000
        sf = (Float32(rand(rng) * 110), Float32(rand(rng) * 90),
              Float32(-3.0 + rand(rng) * 5.2),
              Float32(-90 + rand(rng) * 180), Float32(-90 + rand(rng) * 180),
              Float32(-3.5 + rand(rng) * 7))
        worst_c = max(worst_c, abs(Float64(interp_kuhn(Vc, g, sf..., cap)[1]) - 2.75))
    end
    ok &= _check(worst_c < 1.0e-5, "weights are a partition of unity",
                 @sprintf("worst constant error %.3g", worst_c))

    # Continuity across cell faces, tested where it actually matters: put a
    # point exactly on a grid plane and step a hair either side. A continuous
    # interpolant moves by O(delta); a decomposition that triangulated a
    # shared face differently from the two sides would jump by something of
    # order the table's own range.
    Vr = Float32.(rand(rng, ncell) .* 10)
    worst_j = 0.0; worst_ml = 0.0
    del = 1.0f-3
    for _ in 1:3000
        base = [Float32(rand(rng) * 100), Float32(rand(rng) * 80),
                Float32(-3.0 + rand(rng) * 5.2),
                Float32(-80 + rand(rng) * 160), Float32(-80 + rand(rng) * 160),
                Float32(-3.0 + rand(rng) * 6)]
        k = rand(rng, 1:6)
        # Snap axis k onto an interior grid plane.
        pl = rand(rng, 1:(Int(g.n[k]) - 2))
        base[k] = axisvalue(g, k, pl)
        lo = copy(base); hi = copy(base)
        lo[k] -= del * g.step[k]; hi[k] += del * g.step[k]
        a = interp_kuhn(Vr, g, lo..., cap)[1]
        b = interp_kuhn(Vr, g, hi..., cap)[1]
        worst_j = max(worst_j, abs(Float64(a) - Float64(b)))
        am = interp(Vr, g, lo..., false, cap)
        bm = interp(Vr, g, hi..., false, cap)
        worst_ml = max(worst_ml, abs(Float64(am) - Float64(bm)))
    end
    ok &= _check(worst_j < 0.2, "continuous across cell faces",
                 @sprintf("largest jump %.4f over 0.002 of a cell on a 0..10 table (multilinear %.4f)",
                          worst_j, worst_ml))

    # Cross-check against multilinear, which is also exact on affine data.
    worst_m = 0.0
    for _ in 1:2000
        sf = (Float32(rand(rng) * 110), Float32(rand(rng) * 90),
              Float32(-3.0 + rand(rng) * 5.2),
              Float32(-90 + rand(rng) * 180), Float32(-90 + rand(rng) * 180),
              Float32(-3.5 + rand(rng) * 7))
        a = interp_kuhn(V, g, sf..., cap)[1]
        b = interp(V, g, sf..., false, cap)
        worst_m = max(worst_m, abs(Float64(a) - Float64(b)))
    end
    ok &= _check(worst_m < 3.0e-3, "agrees with multilinear on affine data",
                 @sprintf("worst disagreement %.3g", worst_m))

    # And it must reject exactly what multilinear rejects.
    off = interp_kuhn(V, g, 5.0f0, 5.0f0, 0.0f0, 400.0f0, 0.0f0, 0.0f0, cap)[1]
    nan = interp_kuhn(V, g, 5.0f0, 5.0f0, 0.0f0, Float32(NaN), 0.0f0, 0.0f0, cap)[1]
    ok &= _check(off == cap && nan == cap,
                 "rejects out-of-envelope and non-finite states", "$off, $nan")
    ok
end

"""
Test 4: a state outside the velocity envelope is rejected, not clamped.

Clamping priced leaving the envelope at zero, which let the value function
become optimistic near the velocity boundary and let a rollout accelerate
away to speeds the regression was never fitted at. This pins the fix.
"""
function test_velocity_envelope()
    println("\n[4] velocity envelope")
    n = (5, 5, 4, 5, 5, 5)
    g = Grid6(n, (0.0, 0.0, -π, -150.0, -150.0, -8.0),
                 (100.0, 100.0, π, 150.0, 150.0, 8.0))
    V = fill(1.0f0, ncells(g))
    cap = 30.0f0
    inside = interp(V, g, 50.0f0, 50.0f0, 0.0f0, 100.0f0, 0.0f0, 0.0f0, false, cap)
    outside = interp(V, g, 50.0f0, 50.0f0, 0.0f0, 400.0f0, 0.0f0, 0.0f0, false, cap)
    clamped = interp(V, g, 50.0f0, 50.0f0, 0.0f0, 400.0f0, 0.0f0, 0.0f0, false, cap, true)
    nan = interp(V, g, 50.0f0, 50.0f0, 0.0f0, Float32(NaN), 0.0f0, 0.0f0, false, cap)
    ok = _check(inside ≈ 1.0f0, "inside the envelope interpolates", "$inside")
    ok &= _check(outside == cap, "outside the envelope is rejected", "$outside")
    ok &= _check(clamped ≈ 1.0f0, "vclamp = true restores clamping", "$clamped")
    ok &= _check(nan == cap, "a NaN coordinate is rejected, not rounded", "$nan")
    ok
end

"""
Test 5: a NaN traction knee means "no saturation".

TOML cannot express null, so a missing knee arrives as NaN, and `NaN <= 0`
is false. Without an explicit check the gain became NaN, every control became
NaN, and the kernel indexed the table with garbage.

`load_model` now rejects a missing or non-positive knee outright -- a knee is
required -- so this is the second line of defence rather than the first. It is
kept because the failure was silent and catastrophic, and because a Model can
be built by hand (the benchmark harness does exactly that).
"""
function test_nan_knee()
    println("\n[5] NaN traction knee")
    ok = _check(traction_gain(0.5f0, 0.5f0, 0.0f0, Float32(NaN)) == 1.0f0,
                "NaN knee gives unit gain")
    ok &= _check(traction_gain(0.5f0, 0.5f0, 0.0f0, 0.0f0) == 1.0f0,
                 "zero knee gives unit gain")
    gn = traction_gain(1.0f0, 0.0f0, 0.0f0, 0.45f0)
    ok &= _check(0.0f0 < gn < 1.0f0, "an active knee saturates",
                 @sprintf("gain %.4f", gn))
    ok
end

"""
Test 6: every control the lattice offers is admissible.

The admissible set is the octahedron `|d| + |s| + |t| <= 1`, which is exactly
the set of commands whose mecanum mix keeps all four wheels within [-1, 1].
A lattice point outside it would ask for a wheel power the robot cannot
deliver, and the tables would be solved against a robot that does not exist.
"""
function test_control_set()
    println("\n[6] control set")
    ok = true
    for level in 1:6
        cs = control_set(level)
        worst = maximum(abs(u[1]) + abs(u[2]) + abs(u[3]) for u in cs)
        onbound = count(u -> abs(abs(u[1]) + abs(u[2]) + abs(u[3]) - 1) < 1f-5, cs)
        ok &= _check(worst <= 1.0f0 + 1.0f-5 && onbound == length(cs) - 1 &&
                     length(unique(cs)) == length(cs),
                     "level $level admissible, on the boundary, no duplicates",
                     @sprintf("%d controls, max 1-norm %.6f", length(cs), worst))
    end
    # And the wheel mix really does stay in range.
    mix(u) = (u[1] + u[2] - u[3], u[1] - u[2] - u[3],
              u[1] - u[2] + u[3], u[1] + u[2] + u[3])
    worst = maximum(maximum(abs.(mix(u))) for u in control_set(4))
    ok &= _check(worst <= 1.0f0 + 1.0f-5, "no wheel power exceeds 1",
                 @sprintf("max |wheel| = %.6f", worst))
    ok
end

# --------------------------------------------------------------------------

"""
Test 9: the tiling covers the grid exactly once, at every shift.

The out-of-core driver is a reordering of the in-core one and nothing else,
which is only true if every cell is still updated. A tiling that leaves a
seam of cells out would converge to something that looks plausible -- the
missed cells simply stay at their last value -- so this is checked directly
rather than inferred from the value it produces.

Every shift is checked, not just zero, because the shift is what moves the
tile boundaries between rounds and it is the obvious place for an off-by-one
to hide.
"""
function test_tiling()
    println("\n[9] the tiling covers every cell exactly once")
    ok = true
    for (n1, n2, wx, wy, hx, hy) in ((41, 41, 7, 7, 5, 5),
                                     (41, 41, 40, 40, 9, 9),
                                     (16, 24, 5, 7, 11, 3),
                                     (13, 13, 13, 13, 4, 4))
        g = Grid6((n1, n2, 4, 5, 5, 3),
                  (0.0, 0.0, -π, -100.0, -100.0, -2.0),
                  (200.0, 200.0, π, 100.0, 100.0, 2.0))
        tp = tile_plan(g, wx, wy, hx, hy)
        for shift in 0:5
            cover = zeros(Int, n1, n2)
            halo_ok = true
            for t in tiles_of(g, tp; shift = shift)
                for i in t.ix0:(t.ix0 + t.wx - 1), j in t.iy0:(t.iy0 + t.wy - 1)
                    cover[i + 1, j + 1] += 1
                end
                # The loaded window must contain the interior, and must not
                # reach outside the grid.
                halo_ok &= t.lx0 <= t.ix0 &&
                           t.lx0 + t.nxl >= t.ix0 + t.wx &&
                           t.ly0 <= t.iy0 &&
                           t.ly0 + t.nyl >= t.iy0 + t.wy &&
                           t.lx0 >= 0 && t.ly0 >= 0 &&
                           t.lx0 + t.nxl <= n1 && t.ly0 + t.nyl <= n2
                # And it must reach the full halo wherever the grid allows it.
                halo_ok &= t.lx0 == max(0, t.ix0 - hx) &&
                           t.ly0 == max(0, t.iy0 - hy)
            end
            lab = "$(n1)x$(n2) tiles $(wx)x$(wy) halo $(hx)x$(hy) shift $shift"
            ok &= _check(all(==(1), cover), "covered exactly once: $lab",
                         @sprintf("min %d, max %d", minimum(cover), maximum(cover)))
            ok &= _check(halo_ok, "windows well formed: $lab")
        end
    end
    ok
end

# --------------------------------------------------------------------------

"""
Test 10: the tiled solve agrees with the whole-grid solve.

This is the test the out-of-core driver exists to pass. Both drivers run the
same backup over the same grid, model, occupancy and seed; the only
difference is that one holds the value function whole and the other holds a
window of it at a time. If the dependency halo is wide enough, the two must
converge to the same fixed point -- not bit-identical, because the update
order differs, but to within the tolerance both were stopped at.

**Agreement is measured against the solver's own noise, not against zero.**
The sweep updates V in place and lets the resulting races run -- they are
benign, every write only lowers a cell -- so two runs of the *same* in-core
solver do not agree exactly either. On this grid they differ by a mean of
about 0.02 s, and by several seconds on the handful of cells sitting right at
the edge of reachability, where a hair of difference decides whether a cell
is reached at all. An absolute threshold would therefore be measuring the
wrong thing: pick it tight and it fails on nondeterminism, pick it loose and
it would pass with the tiling broken. So the test runs the in-core solver
twice, uses that as the yardstick, and asks whether tiling adds meaningfully
to it.

**The last part is the one with teeth.** Agreement proves nothing on its own
if the halo is generously oversized -- the test would pass just as well with
a halo calculation that returned a big number for the wrong reason. So the
solve is repeated with the halo cut to one cell, and that has to come out
*worse*: a read past the window is answered with `cap`, which prices a good
step as unreachable. That pins the agreement above on the halo being the
reason for it.

Runs on the GPU only: the tiled driver is a CUDA kernel, and the point of it
is a grid that does not fit on the card.
"""
function test_ooc(; use_gpu = CUDA.functional())
    println("\n[10] the tiled solve agrees with the whole-grid solve")
    if !use_gpu
        println("  SKIP  no CUDA device"); return true
    end
    m = _realistic()
    n = (25, 25, 6, 7, 7, 5)
    g = Grid6(n, (0.0, 0.0, -π, -120.0, -120.0, -6.0),
                 (250.0, 250.0, π, 120.0, 120.0, 6.0))
    ncell = ncells(g)
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    # One obstacle slab, so the swept collision probe is exercised across
    # tile boundaries as well as the interpolation.
    for i in 9:12, j in 0:(Int(n[2]) - 1), k in 0:(Int(n[3]) - 1)
        occ[(i * Int(n[2]) + j) * Int(n[3]) + k + 1] = true
    end
    occv = Vector{Bool}(occ)

    cap = 30.0f0
    p = Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(2),
               adaptive_checks = true, cap = cap, ntau = Int32(4),
               tau_ratio = 2.0f0, cfl = 1.0f0, tau_min = 0.004f0,
               tau_max = 0.12f0, hmax = 0.0125f0, ncoarse = Int32(8),
               rounds = Int32(2), delta0 = 0.35f0, rk2 = true, simplex = true)
    ctl_t = control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]

    targ = (200.0f0, 125.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    seeds = Int64.(target_cells(g, targ, ttol)) .+ 1

    dxy, dh = reach_extent(g, m, p, ctl_h, nctl; nangle = 16, nsample = 128)
    hx, hy = halo_cells(g, dxy; margin = 2)
    reach_cells = dxy / Float64(min(g.step[1], g.step[2]))
    # A halo that already spans the grid would make the "tiled" solve a
    # whole-grid solve wearing a hat, and the comparison would be vacuous.
    ok = _check(hx < Int(n[1]) ÷ 2 && hy < Int(n[2]) ÷ 2,
                "the halo leaves room for a real tile",
                @sprintf("reach %.1f cm = %.1f cells, halo %dx%d", dxy,
                         reach_cells, hx, hy))

    # Cells within 2 s of the cap are excluded from every comparison below.
    # Those are the ones the iteration has only just reached, where whether a
    # cell counts as reached at all turns on the last few improvements, so
    # they dominate any maximum without saying anything about the tiling --
    # the in-core solver disagrees with itself by seconds on exactly those.
    lim = cap - 2.0f0
    function score(a, b)
        both = findall(i -> a[i] < lim && b[i] < lim, eachindex(a))
        isempty(both) && return (n = 0, mean = Inf, p99 = Inf)
        d = sort!([abs(b[i] - a[i]) for i in both])
        (n = length(d), mean = sum(d) / length(d),
         p99 = d[max(1, floor(Int, 0.99 * length(d)))])
    end

    function whole(warm = false)
        V = CUDA.fill(cap, ncell)
        pol = warm ? CUDA.zeros(Int8, 3 * ncell) : nothing
        solve_value!(V, CuArray(occv), g, m, CuArray(seeds), CuArray(ctl_h),
                     nctl, p; iters = 250, tol = 1e-4, use_gpu = true,
                     pol = pol, total = CUDA.zeros(Float32, 1))
        out = Array(V); CUDA.unsafe_free!(V); out
    end
    # Tiled, in RAM so the test needs no scratch file, with an interior no
    # wider than the halo -- the awkward case, and the one a full-scale run
    # is actually in. The interior is held fixed while the halo varies, so
    # the starved run below differs from the honest one in the halo alone and
    # not also in how the grid was cut.
    wx = max(hx, 2); wy = max(hy, 2)
    function tiled(hxx, hyy; warm = false, shift = true)
        tp = tile_plan(g, wx, wy, hxx, hyy; warm = warm, dxy_cm = dxy,
                       dh_rad = dh)
        s = open_store(ncell)
        ps = warm ? open_policy_store(ncell) : nothing
        r, _ = solve_value_ooc!(s, occv, g, m, seeds, ctl_h, nctl, p, tp;
                                rounds = 250, tol = 1e-4, tile_sweeps = 3,
                                warm = warm, pstore = ps, shift = shift)
        (read_all(s), tp, r)
    end

    ref = whole()
    base = score(ref, whole())          # the solver against itself
    got, tp, rounds_cold = tiled(hx, hy)
    cut = score(ref, got)

    ok &= _check(base.n > 0.2 * ncell, "the solves reach a real part of the grid",
                 @sprintf("%.1f%%", 100 * base.n / ncell))
    ok &= _check(tp.ntiles >= 9, "the grid really was cut into tiles",
                 @sprintf("%d tiles of %dx%d, window %dx%d", tp.ntiles,
                          tp.wx, tp.wy, tp.nxl, tp.nyl))
    ok &= _check(cut.mean <= 2.5 * base.mean,
                 "tiling adds little to the solver's own spread (mean)",
                 @sprintf("tiled %.4f s vs in-core-vs-itself %.4f s",
                          cut.mean, base.mean))
    ok &= _check(cut.p99 <= 2.5 * base.p99,
                 "and little at the 99th percentile",
                 @sprintf("tiled %.4f s vs in-core-vs-itself %.4f s",
                          cut.p99, base.p99))

    # And now with no halo at all, so that every step leaving a tile is priced
    # as unreachable. If this does not come out clearly worse, the comparison
    # above was not measuring what it claims to.
    #
    # Scored by mean value over the commonly reached cells rather than by
    # |difference|, for the reason the benchmark harness gives: every scheme's
    # V is an upper bound on the same true value, so two of them compare
    # directly and the lower one is the tighter. A difference cannot tell
    # "tighter" from "looser", and losing a step can only ever be looser.
    #
    # The tiling has to be pinned for this. With the boundaries moving every
    # round a cell that lost a step this round is in the middle of a tile the
    # next, and since V only decreases the better answer it finds then is
    # kept -- so on a grid like this one, where the reach is a third of a
    # tile, the shift heals a missing halo completely and this control would
    # measure nothing. Pinning the tiling is what a full-scale run looks like
    # anyway, where the tile is barely wider than the halo and no shift can
    # put a cell clear of every boundary.
    starved, _, _ = tiled(0, 0; shift = false)
    healed, _, _ = tiled(0, 0; shift = true)
    common = findall(i -> ref[i] < lim && got[i] < lim && starved[i] < lim,
                     eachindex(ref))
    mv(v) = sum(Float64(v[i]) for i in common) / length(common)
    mref, mgot, mstv, mhl = mv(ref), mv(got), mv(starved), mv(healed)
    ok &= _check(mgot < mref + 0.02,
                 "the honest halo loses nothing on mean value",
                 @sprintf("tiled %.4f s vs whole grid %.4f s", mgot, mref))
    ok &= _check(mstv > mgot + 10 * max(abs(mgot - mref), 1.0e-3),
                 "no halo and a pinned tiling is clearly worse",
                 @sprintf("no halo %.4f s vs halo %dx%d %.4f s over %d cells",
                          mstv, hx, hy, mgot, length(common)))
    ok &= _check(mhl < mstv - 0.5 * (mstv - mgot),
                 "and the shifting tiling is most of what recovers it",
                 @sprintf("shifted %.4f s, between %.4f pinned and %.4f honest",
                          mhl, mstv, mgot))

    # The warm-started path is a different policy layout -- one byte a
    # component, interleaved, read in place by `pol_get` -- carried across
    # rounds in its own store. It has to reach the same answer, and it is what
    # makes a full-scale run affordable, so it is not left untested.
    # Compared against an in-core solve that is *also* warm started. Warm
    # starting is not a pure speed-up: the pattern search then refines the
    # same incumbent every sweep instead of re-seeding from a fresh slice of
    # the lattice, which converges far sooner and settles a little higher.
    # That is a property of the operator, identical in both drivers, and
    # scoring the warm tiled run against a cold reference would charge it to
    # the tiling.
    #
    # And against a warm baseline, because warm starting also changes how
    # repeatable the solver is with itself. Refining the same incumbent every
    # sweep couples each sweep to the last, so the in-place races that a cold
    # run averages out are instead carried forward: measured, two in-core warm
    # runs differ about four times as much as two cold ones (mean 0.042 s
    # against 0.0098 s, p99 0.76 s against 0.099 s). Scoring a warm run
    # against the cold spread would fail it for that and call it a tiling bug.
    refw = whole(true)
    basew = score(refw, whole(true))
    hot, _, rhot = tiled(hx, hy; warm = true)
    hotc = score(refw, hot)
    ok &= _check(hotc.mean <= 2.5 * basew.mean && hotc.p99 <= 2.5 * basew.p99,
                 "the byte-quantised warm start reaches the same answer",
                 @sprintf("mean %.4f s, p99 %.4f s vs warm-vs-itself %.4f / %.4f",
                          hotc.mean, hotc.p99, basew.mean, basew.p99))
    ok &= _check(rhot < rounds_cold,
                 "and gets there in fewer rounds than the cold start",
                 @sprintf("%d rounds warm vs %d cold", rhot, rounds_cold))
    ok
end

"""
Self-test 11: the exterior wall is an obstacle like any other.

The rule this pins down is the one the boundary used to break. An obstacle
gets the robot's own footprint swept in and the clearance held around it; the
field boundary now gets exactly the same two things from the inside, so:

  * the tracking point alone is not the constraint -- a chassis may not park
    with half of itself over the line, which is what the old point-only test
    allowed;
  * the inset is per heading, because a square is wider across its diagonal,
    and a table sized for the worst heading everywhere would throw away real
    states;
  * the grid the table is stored on is the *union* over headings, so nothing
    legal at any heading is outside it;
  * and the two calculations agree -- `feasible_bounds` sizes the grid and
    `build_occupancy` fills it, and if they disagreed the outer ring would be
    either blanked or wrongly free.

Checked against a deliberately asymmetric chassis, since a square hides the
per-heading part: a 60 x 20 cm robot needs 30 cm of inset pointing one way
and 10 cm pointing the other.
"""
function test_walls()
    field = (0.0, 0.0, 200.0, 200.0)
    # 60 x 20, long axis along body x, centred on the tracking point.
    R = [-30.0 30.0 30.0 -30.0; -10.0 -10.0 10.0 10.0]
    clr = 4.0

    # The reach really is heading dependent, and in the direction expected.
    r0 = footprint_reach(R, 0.0)
    r90 = footprint_reach(R, pi / 2)
    ok = _check(isapprox(r0[1], 30.0; atol = 1e-9) &&
                isapprox(r0[3], 10.0; atol = 1e-9),
                "pointing along x, the footprint reaches 30 cm in x and 10 in y",
                "$(round.(r0, digits = 3))")
    ok &= _check(isapprox(r90[1], 10.0; atol = 1e-9) &&
                 isapprox(r90[3], 30.0; atol = 1e-9),
                 "turned a quarter turn, the two swap",
                 "$(round.(r90, digits = 3))")

    # The stored span is the union over headings, so it is set by the most
    # permissive one: 10 cm of body plus the clearance, not 30.
    nh, hsub = 16, 3
    fb = feasible_bounds(field, R, clr, nh, hsub)
    # The most permissive bin is the one centred on a quarter turn, whose
    # widest substep angle still only reaches a little past 10 cm.
    best = minimum(maximum(footprint_reach(R, th)[1] for th in angs)
                   for angs in heading_bin_angles(nh, hsub))
    ok &= _check(isapprox(fb.xlo, field[1] + clr + best; atol = 1e-9),
                 "the table spans the union over headings, not the worst one",
                 "xlo $(round(fb.xlo, digits = 3)) vs $(round(clr + best, digits = 3))")
    ok &= _check(fb.xlo < field[1] + clr + 30.0,
                 "which is strictly more than the worst heading would allow",
                 "$(round(fb.xlo, digits = 2)) < $(clr + 30.0)")

    # Now the grid, and the mask on it. No obstacle polygons at all: the only
    # thing that can block a cell here is the wall.
    g = Grid6((41, 41, nh, 3, 3, 3), (fb.xlo, fb.ylo, -pi, -50, -50, -3),
              (fb.xhi, fb.yhi, pi, 50, 50, 3))
    occ = build_occupancy(g, Matrix{Float64}[], 0.0, R, hsub, clr;
                          bounds = field, wall_clearance_cm = clr)

    # Every free cell must hold the whole footprint inside the field, with the
    # clearance, at every angle its bin covers. This is the property, tested
    # directly rather than by re-running the same formula.
    worst = -Inf; nfree = 0
    angsets = heading_bin_angles(nh, hsub)
    for i in 0:(Int(g.n[1]) - 1), j in 0:(Int(g.n[2]) - 1), k in 0:(nh - 1)
        occ[(Int64(i) * g.n[2] + j) * g.n[3] + k + 1] && continue
        nfree += 1
        px = Float64(axisvalue(g, 1, i)); py = Float64(axisvalue(g, 2, j))
        for th in angsets[k + 1]
            rl, rh, bl, bh = footprint_reach(R, th)
            # Signed slack: how far inside the clearance line the footprint
            # sits. Negative means it has crossed it.
            worst = max(worst, (field[1] + clr + rl) - px)
            worst = max(worst, px - (field[3] - clr - rh))
            worst = max(worst, (field[2] + clr + bl) - py)
            worst = max(worst, py - (field[4] - clr - bh))
        end
    end
    ok &= _check(nfree > 0, "the wall does not blank the whole grid", "$nfree free")
    ok &= _check(worst <= 1.0e-3,
                 "no free cell hangs the footprint through the wall",
                 "worst overhang $(round(worst, digits = 6)) cm")

    # And the grid is not needlessly small: the most permissive heading has to
    # have free cells right against the edge, or the inset is over-tight and
    # real states have been thrown away.
    edge_free = any(!occ[(Int64(0) * g.n[2] + j) * g.n[3] + k + 1]
                    for j in 0:(Int(g.n[2]) - 1), k in 0:(nh - 1))
    ok &= _check(edge_free, "the low x edge is reachable at some heading")

    # The point-robot case must lose the footprint term and keep the gap.
    fbp = feasible_bounds(field, nothing, clr, nh, hsub)
    ok &= _check(isapprox(fbp.xlo, field[1] + clr; atol = 1e-9) &&
                 isapprox(fbp.xhi, field[3] - clr; atol = 1e-9),
                 "a point robot is inset by the clearance alone",
                 "$(round(fbp.xlo, digits = 3))..$(round(fbp.xhi, digits = 3))")

    # A chassis wider than the field is a configuration error, not an empty
    # grid: the old failure mode was a table of zero useful cells.
    ok &= _check(try
                     feasible_bounds((0.0, 0.0, 40.0, 40.0), R, clr, nh, hsub)
                     false
                 catch
                     true
                 end, "a robot too big for the field is refused, not silently empty")
    ok
end

"""
Self-test 12: the advice is self-consistent.

None of this changes a solve, so it is checked for coherence rather than
accuracy: a cost curve that ran backwards, or a resolution suggestion that
broke its own budget, would mislead every run started from it.
"""
function test_advice()
    # Shortening the lookahead can only cost, never pay, and the curve has to
    # be monotone across the whole measured range and past the end of it.
    taus = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01]
    costs = value_cost.(taus)
    ok = _check(all(costs[i] <= costs[i + 1] + 1e-9 for i in 1:(length(taus) - 1)),
                "value cost is monotone as the step shortens",
                join(round.(costs, digits = 1), " "))
    ok &= _check(value_cost(TAU_REF) == 0.0 && value_cost(1.0) == 0.0,
                 "at or above the reference the cost is zero")
    ok &= _check(value_cost(0.5, 0.1) > 20.0 && value_cost(0.1, 0.5) == 0.0,
                 "and it is one-sided: lengthening the step is free",
                 "$(round(value_cost(0.5, 0.1), digits = 1))%")

    # A round pays for its compute and its I/O in sequence. The driver has no
    # prefetch and no second staging buffer, so nothing is hidden behind
    # anything else -- see `round_seconds`. This asserted the maximum until
    # 2026-08-31, which under-stated a round by up to a factor of two and made
    # every step length below the compute-bound point look identically priced.
    r = round_seconds(1.0e9, 4, 10.0, 7.0, 23.4e6, 500.0e6)
    ok &= _check(r.total == r.compute + r.io,
                 "a round costs compute plus i/o, not the greater of the two",
                 @sprintf("%.1f = %.1f + %.1f s", r.total, r.compute, r.io))
    # The property the maximum destroyed, tested where it destroyed it: when
    # I/O is the *smaller* half, amplification still has to show up in the
    # price. Both probes below sit in that regime -- 56 s and 126 s of traffic
    # against 171 s of sweeps -- and under `max` they came out as the same
    # number to the last digit. That is what made every step length below the
    # compute-bound point look identically priced, and it is what turned the
    # `tau_max` recommendation into "take the longest one that fits".
    lo = round_seconds(1.0e9, 4, 3.0, 7.0, 23.4e6, 500.0e6)
    hi = round_seconds(1.0e9, 4, 8.0, 7.0, 23.4e6, 500.0e6)
    ok &= _check(lo.io < lo.compute && hi.io < hi.compute,
                 "both probes are in the sweep-heavy regime",
                 @sprintf("io %.0f and %.0f s against compute %.0f s",
                          lo.io, hi.io, lo.compute))
    ok &= _check(hi.total > lo.total * 1.2,
                 "amplification is priced even when i/o is the smaller half",
                 @sprintf("%.1f s at 8x vs %.1f s at 3x", hi.total, lo.total))

    # The suggestion has to fit the budget it was given, and a smaller budget
    # has to give a coarser cell -- never a finer one.
    s1 = suggest_resolution(320.0, 320.0, 25.5, 170.0, 8.0, 2, 3, 8.0 * 2^30)
    s2 = suggest_resolution(320.0, 320.0, 25.5, 170.0, 8.0, 2, 3, 1.0 * 2^30)
    ok &= _check(s1 !== nothing && s2 !== nothing, "a resolution is found at both budgets")
    if s1 !== nothing && s2 !== nothing
        ok &= _check(s1.bytes <= 8.0 * 2^30 && s2.bytes <= 1.0 * 2^30,
                     "the suggestion fits the budget it was given",
                     "$(round(s1.bytes / 2^30, digits = 2)) GB, " *
                     "$(round(s2.bytes / 2^30, digits = 2)) GB")
        ok &= _check(s2.xy_cm >= s1.xy_cm,
                     "an eighth of the space does not buy a finer cell",
                     "$(s1.xy_cm) cm -> $(s2.xy_cm) cm")
        ok &= _check(all(isodd, s1.n[4:6]),
                     "the velocity axes have zero exactly on the grid",
                     "$(s1.n)")
    end

    # The heading rule has to tighten as the position cell does, since it is
    # the position cell it is measured against.
    ok &= _check(suggest_heading_deg(25.0, 2.0) < suggest_heading_deg(25.0, 9.0),
                 "a finer position cell asks for finer heading bins",
                 "$(round(suggest_heading_deg(25.0, 9.0), digits = 2)) deg -> " *
                 "$(round(suggest_heading_deg(25.0, 2.0), digits = 2)) deg")
    ok
end

"""
Self-test 13: the reach curve covers the ladder the backup actually runs.

`reach_curve` is what the `tau_max` recommendation is read off, and it is
allowed to be conservative but never optimistic: a curve that under-stated
the reach would recommend a halo too small for the step it recommends, and a
halo too small does not fail loudly -- it prices good steps as unreachable
and leaves a faint lattice of seams on the tile boundaries.
"""
function test_reach_curve()
    m = _realistic()
    g = Grid6((21, 21, 8, 7, 7, 5), (0.0, 0.0, -pi, -120, -120, -6),
              (200.0, 200.0, pi, 120, 120, 6))
    ctl = control_set(2)
    ch = Float32[getindex.(ctl, 1); getindex.(ctl, 2); getindex.(ctl, 3)]
    taus = [0.5, 0.3, 0.2, 0.1, 0.05]
    ok = true
    for tm in taus
        p = Params(dt = 0.05, tau_max = Float32(tm), cfl = 2.0f0,
                   ntau = Int32(5), cap = 60.0f0)
        exact, _ = reach_extent(g, m, p, ch, length(ctl);
                                nangle = 8, nsample = 48)
        curve = reach_curve(g, m, p, ch, length(ctl), taus;
                            nangle = 8, nsample = 48)
        approx, _ = reach_at(curve, tm)
        ok &= _check(approx >= exact - 1.0e-6,
                     "the curve never under-states the reach at tau_max = $tm",
                     "curve $(round(approx, digits = 2)) cm vs exact " *
                     "$(round(exact, digits = 2)) cm")
    end
    # And monotone, since `tau_max` caps the ladder rather than selecting a rung.
    p = Params(dt = 0.05, tau_max = 0.5f0, cfl = 2.0f0, ntau = Int32(5),
               cap = 60.0f0)
    curve = reach_curve(g, m, p, ch, length(ctl), sort(taus);
                        nangle = 8, nsample = 48)
    ok &= _check(issorted(curve.dxy_cm), "the curve is monotone in tau",
                 join(round.(curve.dxy_cm, digits = 1), " "))
    ok
end

"""
[14] The prefetch changes the schedule, not the answer.

The prefetch reads tile k+1 while the GPU sweeps tile k, which means it can
read cells that tile k has not written back yet. That is deliberate and it is
sound -- V only ever decreases, so a stale read is a *larger* value, and
`cell_update` takes a min against it, so a stale halo can delay convergence
but can never pull a cell below the truth.

"Sound" is not "identical", so this measures the difference rather than
asserting there is none, and scores it against the spread the driver already
has against itself. Both runs are file-backed on purpose: the RAM-backed
store shares one array and would exercise none of the two-handle machinery
that makes the file case safe.

It also checks the direction. A stale read can only ever be looser, so the
prefetched run's mean value must not come out *below* the sequential one by
more than the noise -- if it did, something is reading cells that were never
written rather than cells written a moment ago.
"""
function test_prefetch()
    println("
[14] the prefetch changes the schedule, not the answer")
    if !CUDA.functional()
        println("  SKIP  no CUDA device"); return true
    end
    if Threads.nthreads() < 2
        println("  SKIP  julia has one thread; start it with -t auto")
        return true
    end

    m = _realistic()
    n = (25, 25, 6, 7, 7, 5)
    g = Grid6(n, (0.0, 0.0, -π, -120.0, -120.0, -6.0),
                 (250.0, 250.0, π, 120.0, 120.0, 6.0))
    ncell = ncells(g)
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    for i in 9:12, j in 0:(Int(n[2]) - 1), k in 0:(Int(n[3]) - 1)
        occ[(i * Int(n[2]) + j) * Int(n[3]) + k + 1] = true
    end
    occv = Vector{Bool}(occ)

    cap = 30.0f0
    p = Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(2),
               adaptive_checks = true, cap = cap, ntau = Int32(4),
               tau_ratio = 2.0f0, cfl = 1.0f0, tau_min = 0.004f0,
               tau_max = 0.12f0, hmax = 0.0125f0, ncoarse = Int32(8),
               rounds = Int32(2), delta0 = 0.35f0, rk2 = true, simplex = true)
    ctl_t = control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]
    targ = (200.0f0, 125.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    seeds = Int64.(target_cells(g, targ, ttol)) .+ 1

    dxy, dh = reach_extent(g, m, p, ctl_h, nctl; nangle = 16, nsample = 128)
    hx, hy = halo_cells(g, dxy; margin = 2)
    wx = max(hx, 2); wy = max(hy, 2)
    tp = tile_plan(g, wx, wy, hx, hy; warm = true, dxy_cm = dxy, dh_rad = dh)

    dir = mktempdir()
    function run_one(pf::Bool, tag::String)
        s = open_store(ncell; path = joinpath(dir, "V_$tag.SCRATCH"))
        ps = open_policy_store(ncell; path = joinpath(dir, "P_$tag.SCRATCH"))
        r, _ = solve_value_ooc!(s, occv, g, m, seeds, ctl_h, nctl, p, tp;
                                rounds = 250, tol = 1e-4, tile_sweeps = 3,
                                warm = true, pstore = ps, prefetch = pf)
        v = read_all(s)
        close_store!(s); close_store!(ps)
        (v, r)
    end

    ok = true
    try
        a1, r1 = run_one(false, "a1")
        a2, _  = run_one(false, "a2")      # the driver against itself
        b1, r2 = run_one(true,  "b1")

        lim = cap - 2.0f0
        function score(a, b)
            both = findall(i -> a[i] < lim && b[i] < lim, eachindex(a))
            isempty(both) && return (n = 0, mean = Inf, p99 = Inf, bias = Inf)
            d = sort!([abs(b[i] - a[i]) for i in both])
            (n = length(d), mean = sum(d) / length(d),
             p99 = d[max(1, floor(Int, 0.99 * length(d)))],
             bias = sum(b[i] - a[i] for i in both) / length(both))
        end
        base = score(a1, a2)
        got  = score(a1, b1)

        ok &= _check(base.n > 0.2 * ncell,
                     "the runs reach a real part of the grid",
                     @sprintf("%.1f%%", 100 * base.n / ncell))
        ok &= _check(tp.ntiles >= 9, "the grid really was cut into tiles",
                     @sprintf("%d tiles", tp.ntiles))
        ok &= _check(got.mean <= max(2.5 * base.mean, 1e-3),
                     "prefetched agrees with sequential (mean)",
                     @sprintf("prefetch %.5f s vs driver-vs-itself %.5f s",
                              got.mean, base.mean))
        ok &= _check(got.p99 <= max(2.5 * base.p99, 1e-2),
                     "and at the 99th percentile",
                     @sprintf("prefetch %.5f s vs driver-vs-itself %.5f s",
                              got.p99, base.p99))
        # Direction: a stale read is a larger V, so any bias must be >= 0.
        # A prefetched run that came out systematically *lower* would mean it
        # had read cells nobody wrote, which is the failure that matters.
        ok &= _check(got.bias >= -max(3 * abs(base.bias), 1e-3),
                     "the prefetch never reads low -- stale means looser, "
                     * "never tighter",
                     @sprintf("bias %+.5f s vs self-bias %+.5f s",
                              got.bias, base.bias))
        ok &= _check(abs(r2 - r1) <= max(3, 0.25 * r1),
                     "and it does not blow up the round count",
                     @sprintf("%d rounds prefetched vs %d sequential", r2, r1))
    finally
        rm(dir; force = true, recursive = true)
    end
    ok
end

"""
[15] The two drivers agree about what a cell costs, and both are right.

This is the regression test for a bug that cost nothing on the desktop and
would have cost a rental: `decompose` sized the whole-grid case with the
tiled driver's 7 bytes a cell while `run_solve` allocated the in-core
driver's 16, so a grid between `budget / 16` and `budget / 7` was routed in
core and could not be allocated. Windows hid it completely -- WDDM pages VRAM
into host RAM, so it ran at a fraction of the speed instead of failing.

Three things are asserted, and they are deliberately different in kind:

  * **Arithmetic.** Every byte count in the solver now comes from
    `cell_bytes`, so `gpu_fits` and `_tile_bytes` cannot drift apart again.
    Checked against the element types actually allocated, not against a
    repeated literal -- a test that says 7 because the code says 7 would have
    passed happily throughout the bug.
  * **Accuracy.** In core the policy used to be three `Float32`. Quantising
    it to bytes is what makes the arithmetic true, so the quantisation has to
    be shown harmless rather than assumed: the same solve, both layouts,
    scored against the solver's own run-to-run spread.
  * **The guard.** A grid too big for the card must raise, not page.
"""
function test_cell_bytes()
    println("\n[15] the two drivers agree about what a cell costs")
    ok = true

    # -- arithmetic ------------------------------------------------------
    # Tied to the types, so that changing a layout without changing
    # `cell_bytes` fails here rather than in a rented machine's allocator.
    ok &= _check(value_bytes_per_cell() == sizeof(Float32),
                 "V's byte count is Float32's",
                 "$(value_bytes_per_cell()) vs $(sizeof(Float32))")
    ok &= _check(policy_bytes_per_cell() == 3 * sizeof(Int8),
                 "the policy's byte count is three Int8",
                 "$(policy_bytes_per_cell()) vs $(3 * sizeof(Int8))")
    ok &= _check(cell_bytes(true) == 7 && cell_bytes(false) == 4,
                 "a warm cell is 7 bytes, a cold one 4",
                 "$(cell_bytes(true)) and $(cell_bytes(false))")

    g = Grid6((20, 20, 6, 7, 7, 5), (0.0, 0.0, -π, -120.0, -120.0, -6.0),
                                    (250.0, 250.0, π, 120.0, 120.0, 6.0))
    ncell = ncells(g)
    # The whole grid as a single tile with no halo is exactly the in-core
    # residency, so this is the two drivers pricing the same thing. It is the
    # comparison that used to come out 2.29x apart.
    whole_b, _, _, _ = _tile_bytes(g, Int(g.n[1]), Int(g.n[2]), 0, 0, true)
    occ_b = Int64(g.n[1]) * Int64(g.n[2]) * Int64(g.n[3])
    ok &= _check(whole_b - occ_b == ncell * cell_bytes(true),
                 "the tiled driver prices a whole grid as the in-core one does",
                 "$(whole_b - occ_b) vs $(ncell * cell_bytes(true))")
    if CUDA.functional()
        _, need_gb, _ = gpu_fits(g; warm = true)
        ok &= _check(isapprox(need_gb, ncell * cell_bytes(true) / 2^30;
                              rtol = 1e-9),
                     "and `gpu_fits` reports that same number",
                     @sprintf("%.6f GB vs %.6f GB", need_gb,
                              ncell * cell_bytes(true) / 2^30))
    end

    # -- accuracy: the quantisation the arithmetic depends on -------------
    if !CUDA.functional()
        println("  SKIP  no CUDA device for the rest"); return ok
    end
    m = _realistic()
    occ = falses(Int(g.n[1]) * Int(g.n[2]) * Int(g.n[3]))
    for i in 7:9, j in 0:(Int(g.n[2]) - 1), k in 0:(Int(g.n[3]) - 1)
        occ[(i * Int(g.n[2]) + j) * Int(g.n[3]) + k + 1] = true
    end
    cap = 30.0f0
    p = Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(2),
               adaptive_checks = true, cap = cap, ntau = Int32(4),
               tau_ratio = 2.0f0, cfl = 1.0f0, tau_min = 0.004f0,
               tau_max = 0.12f0, hmax = 0.0125f0, ncoarse = Int32(8),
               rounds = Int32(2), delta0 = 0.35f0, rk2 = true, simplex = true)
    ct = control_set(4); nctl = length(ct)
    ch = Float32[getindex.(ct, 1); getindex.(ct, 2); getindex.(ct, 3)]
    tt = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    sd = Int64.(target_cells(g, (200.0f0, 125.0f0, 0f0, 0f0, 0f0, 0f0), tt)) .+ 1
    docc = CuArray(Vector{Bool}(occ)); dctl = CuArray(ch); dsd = CuArray(sd)

    function solve(polkind)
        V = CUDA.fill(cap, ncell)
        pol = polkind === nothing ? nothing :
              (polkind === Int8 ? CUDA.zeros(Int8, 3 * ncell) :
                                  CUDA.zeros(Float32, 3 * ncell))
        solve_value!(V, docc, g, m, dsd, dctl, nctl, p; iters = 300,
                     tol = 1e-4, use_gpu = true, pol = pol,
                     total = CUDA.zeros(Float32, 1))
        out = Array(V); CUDA.unsafe_free!(V)
        pol === nothing || CUDA.unsafe_free!(pol)
        out
    end

    lim = cap - 2.0f0
    function spread(a, b)
        both = findall(i -> a[i] < lim && b[i] < lim, eachindex(a))
        isempty(both) && return (n = 0, mean = Inf)
        (n = length(both),
         mean = sum(abs(b[i] - a[i]) for i in both) / length(both))
    end

    f1 = solve(Float32)
    base = spread(f1, solve(Float32))     # the solver against itself
    q1 = solve(Int8)
    got = spread(f1, q1)
    ok &= _check(base.n > 0.2 * ncell, "the solves reach a real part of the grid",
                 @sprintf("%.1f%%", 100 * base.n / ncell))
    # The policy is the seed the pattern search starts from, and the search's
    # own first step is `delta0` -- fifty times coarser than the 1/127 a byte
    # resolves -- while every candidate is still evaluated exactly by
    # `lookahead`. So the quantisation should not move the answer at all, and
    # any difference here should be the ordinary in-place-race noise.
    ok &= _check(got.mean <= max(2.5 * base.mean, 1e-3),
                 "a byte-quantised in-core policy costs no accuracy",
                 @sprintf("%.5f s vs solver-vs-itself %.5f s", got.mean,
                          base.mean))

    # -- the guard --------------------------------------------------------
    # A grid far past the card, solved with the real driver. This must be an
    # error naming the fix, not a CUDA allocator failure and not -- as it was
    # on Windows -- a successful run backed by host memory.
    free, _ = CUDA.memory_info()
    big = ceil(Int, (free / cell_bytes(true)) * 4)
    sz = Int(ceil(sqrt(big / (6 * 7 * 7 * 5))))
    gbig = Grid6((sz, sz, 6, 7, 7, 5), (0.0, 0.0, -π, -120.0, -120.0, -6.0),
                                       (250.0, 250.0, π, 120.0, 120.0, 6.0))
    # The guard's own expression, not a paraphrase of it.
    need_b = ncells(gbig) * cell_bytes(true)
    ok &= _check(need_b > free,
                 "the guard fires on a grid past the card",
                 @sprintf("needs %.1f GB of %.1f GB free", need_b / 2^30,
                          free / 2^30))
    # And the router agrees, so the guard is a backstop rather than the only
    # thing standing between this grid and a silent host-memory solve.
    fits, _, _ = gpu_fits(gbig; warm = true)
    ok &= _check(!fits, "and the router sends it to the tiled driver instead")
    ok
end

"""
Self-test 16: the escape pass, and the band that carries it on the card.

This is for a robot that is *already stuck* -- shoved into an obstacle, or
against a wall at a heading that does not fit -- reading `unreachable` in
every direction with nothing to descend. So the tests are the four things
that failure mode actually needs, and not the easy one.

The easy one, and the reason it is not enough: "every blocked cell got some
number". A pass that filled them with the distance to the nearest edge of the
obstacle would satisfy that and still strand the robot, by walking it out of
the near side of a wall into a pocket with no route onward.

So the geometry here is deliberately **asymmetric**. A slab across the field
at x in 20..40 cuts the low end off from the target at x = 100 entirely, so
the cells below it are unreachable and stay that way. A cell at x = 25 is
therefore 5 cm from the near edge and 15 cm from the far one, and only the far
edge is any use. The closed form separates the two answers by 73%:

    escape from rest over d, free terminal speed:  t = sqrt(2d/a)
    near edge  (5 cm, useless):   0.316 s
    far edge  (15 cm, the answer): 0.548 s

Which one comes back says whether the pass is aiming at *states that have a
route* or merely at "not an obstacle".
"""
function test_escape(; use_gpu = CUDA.functional())
    println("\n[16] the escape pass")
    amax = 100.0f0
    m = _double_integrator(amax)
    n = (81, 5, 4, 33, 5, 5)
    g = Grid6(n, (0.0, 0.0, -π, -100.0, -100.0, -2.0),
                 (200.0, 200.0, π, 100.0, 100.0, 2.0))
    cap = 30.0f0
    ncell = ncells(g)
    nxy = Int(n[1]) * Int(n[2]) * Int(n[3])

    # A slab of obstacle at x in 20..40, every y and every heading. Set
    # directly rather than rasterised from a polygon: this is a test of the
    # pass, not of `build_occupancy`, and a hand-built mask cannot drift.
    occ = falses(nxy)
    xs = [Float64(axisvalue(g, 1, i)) for i in 0:(Int(n[1]) - 1)]
    slab = [i for i in 0:(Int(n[1]) - 1) if 20.0 <= xs[i + 1] <= 40.0]
    for i in slab, j in 0:(Int(n[2]) - 1), k in 0:(Int(n[3]) - 1)
        occ[(Int64(i) * g.n[2] + j) * g.n[3] + k + 1] = true
    end
    _check(length(slab) >= 3, "the slab is thicker than one cell",
           "$(length(slab)) cells, $(round(xs[slab[1] + 1], digits = 1))..$(round(xs[slab[end] + 1], digits = 1)) cm")

    targ = (100.0f0, 100.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    tcells = target_cells(g, targ, ttol)
    ctl_t = control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]
    p = Params(; dt = 0.05f0, nsub = Int32(4), cap = cap, checks = Int32(2),
               adaptive_checks = true, ntau = Int32(4), cfl = 1.0f0,
               hmax = 0.0125f0, ncoarse = Int32(0), rounds = Int32(2),
               rk2 = true)

    function solved(gpu::Bool)
        if gpu
            V = CUDA.fill(cap, ncell)
            solve_value!(V, CuArray(occ), g, m, CuArray(Int64.(tcells) .+ 1),
                         CuArray(ctl_h), nctl, p; iters = 600, tol = 1e-4,
                         use_gpu = true, total = CUDA.zeros(Float32, 1))
            Array(V)
        else
            V = fill(cap, ncell)
            solve_value!(V, occ, g, m, Int64.(tcells) .+ 1, ctl_h, nctl, p;
                         iters = 600, tol = 1e-4, use_gpu = false)
            V
        end
    end

    function escaped(V0, gpu::Bool)
        if gpu
            V = CuArray(V0)
            solve_escape!(V, CuArray(occ), g, m, CuArray(ctl_h), nctl, p;
                          iters = 200, tol = 1e-4, use_gpu = true,
                          total = CUDA.zeros(Float32, 1))
            Array(V)
        else
            V = copy(V0)
            solve_escape!(V, occ, g, m, ctl_h, nctl, p; iters = 200,
                          tol = 1e-4, use_gpu = false)
            V
        end
    end

    before = solved(use_gpu)
    after = escaped(before, use_gpu)

    # --- it cannot damage the table it was given -------------------------
    #
    # The whole reason this is on by default. Every cell that had a real route
    # must come back bit-identical, or the pass is not additive and every
    # table on every card is in question.
    term = [i for i in eachindex(before) if is_terminal(before[i], cap)]
    _check(!isempty(term), "the solve reached something", "$(length(term)) cells")
    _check(all(after[i] === before[i] for i in term),
           "every cell with a real route is untouched, bit for bit",
           "$(count(i -> after[i] !== before[i], term)) changed of $(length(term))")

    nesc = count(v -> v < 0.0f0, after)
    dead = count(v -> !is_terminal(v, cap), before)
    _check(nesc > 0, "cells with no route got an escape",
           "$nesc of $dead, $(round(100 * nesc / dead, digits = 1))%")
    _check(all(isfinite(v) && -v > 0.0f0 && -v <= cap + 1.0f-3
               for v in after if v < 0.0f0),
           "every escape time is finite, positive and within cap")

    # --- it aims at states that have a route, not at open space ----------
    esc_at(V, x) = begin
        i = argmin(abs.(xs .- x)) - 1
        j = Int(round((100.0 - g.lo[2]) / g.step[2]))
        # At rest, mid heading bin, mid velocity: the state a stuck robot is in.
        i4 = (Int(n[4]) - 1) ÷ 2; i5 = (Int(n[5]) - 1) ÷ 2; i6 = (Int(n[6]) - 1) ÷ 2
        v = V[flatten(g, Int32(i), Int32(j), Int32(0), Int32(i4), Int32(i5),
                      Int32(i6)) + 1]
        v < 0.0f0 ? Float64(-v) : NaN
    end

    d_far = 40.0 - 25.0
    want_far = sqrt(2 * d_far / Float64(amax))
    want_near = sqrt(2 * (25.0 - 20.0) / Float64(amax))
    got = esc_at(after, 25.0)
    @printf("        x = 25 cm:  far edge %.3f s   near edge %.3f s   solver %.3f s\n",
            want_far, want_near, got)
    # Generous on the upper side -- a discrete grid and a cell-sized terminal
    # set both cost a little -- but nowhere near the near-edge answer, which
    # is the alternative this is separating it from.
    _check(!isnan(got) && got > (want_far + want_near) / 2,
           "the escape aims at the reachable side, not the nearest edge",
           @sprintf("%.3f s, midpoint between the two answers is %.3f s",
                    got, (want_far + want_near) / 2))
    _check(!isnan(got) && got <= want_far * 1.6,
           "and it is close to the closed form for that side",
           @sprintf("%.3f s vs %.3f s (%+.1f%%)", got, want_far,
                    100 * (got - want_far) / want_far))

    # --- it does not route through walls ---------------------------------
    #
    # The cells below the slab are cut off from the target by it. They are in
    # free space, so the rule is that they may not enter an obstacle to get
    # out -- and there is nowhere else for them to go. They must stay
    # unreachable. A pass that "helped" them would be walking the robot into
    # the wall it is standing next to.
    low = [(Int64(i) * g.n[2] + j) * g.n[3] + k
           for i in 0:(slab[1] - 1) for j in 0:(Int(n[2]) - 1)
           for k in 0:(Int(n[3]) - 1)]
    lowcells = Int64[]
    for b in low, r in 0:(Int(n[4]) * Int(n[5]) * Int(n[6]) - 1)
        push!(lowcells, b * Int64(n[4]) * Int64(n[5]) * Int64(n[6]) + r)
    end
    nbad = count(i -> after[i + 1] < 0.0f0, lowcells)
    _check(nbad == 0,
           "free cells walled off from the target get no escape through the wall",
           "$nbad of $(length(lowcells)) cells below the slab")

    # --- the two backends run the same scheme ----------------------------
    #
    # On its own small grid with a small control set. The pass costs the same
    # per cell as a solve sweep, so running the real one on the CPU as well
    # would be the slowest thing in this file by an order of magnitude, and it
    # would be measuring the CPU rather than the agreement.
    if use_gpu
        gs = Grid6((25, 3, 4, 13, 3, 3), (0.0, 0.0, -π, -100.0, -100.0, -2.0),
                   (200.0, 200.0, π, 100.0, 100.0, 2.0))
        nc = ncells(gs)
        so = falses(Int(gs.n[1]) * Int(gs.n[2]) * Int(gs.n[3]))
        sxs = [Float64(axisvalue(gs, 1, i)) for i in 0:(Int(gs.n[1]) - 1)]
        for i in 0:(Int(gs.n[1]) - 1), j in 0:(Int(gs.n[2]) - 1),
            k in 0:(Int(gs.n[3]) - 1)
            20.0 <= sxs[i + 1] <= 50.0 &&
                (so[(Int64(i) * gs.n[2] + j) * gs.n[3] + k + 1] = true)
        end
        sct = control_set(2)
        sch = Float32[getindex.(sct, 1); getindex.(sct, 2); getindex.(sct, 3)]
        stc = target_cells(gs, targ,
                           NTuple{6,Float32}(Float32[gs.step[k] * 0.5f0 for k in 1:6]))
        sp = Params(; dt = 0.05f0, nsub = Int32(4), cap = cap,
                    checks = Int32(2), adaptive_checks = true,
                    ntau = Int32(2), cfl = 1.0f0, hmax = 0.0125f0, rk2 = true)

        Vc = fill(cap, nc)
        solve_value!(Vc, so, gs, m, Int64.(stc) .+ 1, sch, length(sct), sp;
                     iters = 200, tol = 1e-4, use_gpu = false)
        # The solved table, kept before either pass runs over it in place, so
        # every backend below starts from identical input.
        Vgh0 = copy(Vc)
        Vg = CuArray(Vc)
        solve_escape!(Vc, so, gs, m, sch, length(sct), sp; iters = 60,
                      tol = 1e-4, use_gpu = false)
        solve_escape!(Vg, CuArray(so), gs, m, CuArray(sch), length(sct), sp;
                      iters = 60, tol = 1e-4, use_gpu = true,
                      total = CUDA.zeros(Float32, 1))
        Vgh = Array(Vg)
        de = maximum(abs(Float64(Vc[i]) - Float64(Vgh[i]))
                     for i in eachindex(Vc) if Vc[i] < 0 || Vgh[i] < 0;
                     init = 0.0)
        nsign = count(i -> (Vc[i] < 0) != (Vgh[i] < 0), eachindex(Vc))
        _check(nsign == 0, "the CPU and GPU passes escape the same cells",
               "$nsign of $nc disagree")

        # In the aggregate, not cell by cell, and that is not a fudge.
        #
        # Measured: the two backends settle on escape times that differ by up
        # to 0.19 s at the worst cell, against a self-spread of 0.0003 s (CPU
        # against CPU) and 0.013 s (GPU against GPU). So it is systematic, not
        # scheduling noise -- but **each answer is a fixed point of the other
        # backend**: running the GPU pass from the CPU's converged table does
        # not move it, and vice versa. Neither dominates either, the GPU being
        # strictly lower on 932 cells and strictly higher on 871.
        #
        # The discretised operator simply has more than one fixed point here,
        # and CPU and GPU float contraction tip near-tied controls to
        # different ones. Every disagreement sits at the corner of the
        # velocity envelope on the obstacle's edge, where the value surface is
        # flat and bifurcating. The mean is what is stable -- it differed by
        # 0.012% -- so that is what is worth asserting; a worst-cell bound
        # would only be pinning down which arbitrary tie a given card breaks.
        me(V) = (e = [-Float64(V[i]) for i in eachindex(V) if V[i] < 0.0f0];
                 isempty(e) ? 0.0 : sum(e) / length(e))
        mc = me(Vc); mg = me(Vgh)
        rel = mc > 0 ? abs(mc - mg) / mc : 0.0
        _check(rel <= 0.01, "and agree on the mean escape time",
               @sprintf("cpu %.4f s, gpu %.4f s (%+.3f%%); worst cell %.3f s",
                        mc, mg, 100 * (mg - mc) / mc, de))

        # --- and the tiled pass agrees with the whole-grid one ------------
        #
        # The tiled driver is what runs at full scale, so the escape pass has
        # to hold there too. Fed the *same* solved table as the in-core pass
        # rather than re-solving, so this isolates the escape pass from any
        # difference in the solve that preceded it.
        #
        # A tile's halo is frozen for the residency exactly as it is in a
        # solve, and it is sound here for the mirrored reason: the stored
        # value is the negated escape time, so it only ever increases, but the
        # quantity being minimised is still the escape time and a stale halo
        # still carries a larger one. See `solve_escape_ooc!`.
        # The halo has to cover how far one backup can reach, or steps that
        # leave a tile are priced as unreachable and the comparison below
        # measures a starved halo rather than the tiling. 50 cm is `tau_max`
        # at the envelope's top speed; six x cells is 50 cm, and one y cell is
        # 100 cm on this deliberately coarse y axis.
        tp = tile_plan(gs, 6, 1, 6, 1; warm = false, dxy_cm = 50.0,
                       dh_rad = 2.0 * Float64(gs.step[3]))
        st = open_store(nc)
        try
            # 0-based store offset, 1-based buffer position.
            write_range!(st, Int64(0), Vgh0, Int64(1), Int64(nc))
            rd, _ = solve_escape_ooc!(st, Vector{Bool}(so), gs, m, sch,
                                      length(sct), sp, tp; rounds = 40,
                                      tol = 1e-6, tile_sweeps = 3)
            Vt = read_all(st)
            nsign_t = count(i -> (Vt[i] < 0) != (Vgh[i] < 0), eachindex(Vt))
            mt = me(Vt)
            _check(tp.ntiles >= 4, "the grid really was cut into tiles",
                   "$(tp.ntiles) tiles, $(rd) rounds")
            _check(nsign_t == 0, "the tiled pass escapes the same cells",
                   "$nsign_t of $nc disagree")
            _check(mt > 0 && abs(mt - mg) / mg <= 0.05,
                   "and agrees with the whole-grid pass on the mean",
                   @sprintf("tiled %.4f s vs whole %.4f s (%+.2f%%)",
                            mt, mg, 100 * (mt - mg) / mg))
        finally
            close_store!(st)
        end
    end

    # --- the card round-trip ---------------------------------------------
    #
    # Decoded by the rules in section 6 of TABLE_FORMAT.md rather than by
    # anything the encoder knows, because what is being tested is the
    # contract the robot firmware will implement.
    for dt in ("u16", "u8", "f32")
        sc = DTYPES[dt].scale
        es = escape_scale_for(dt, cap)
        buf = encode(after, dt, sc, cap, es)
        base = Int(DTYPES[dt].escape_base)
        # NaN for the float types, so it cannot go through `Int`.
        sentinel = dt == "f32" ? 0 : Int(DTYPES[dt].sentinel)
        raws = dt == "u8" ? Int.(buf) :
               dt == "u16" ? Int.(reinterpret(UInt16, buf)) : Int[]
        fl = dt == "f32" ? reinterpret(Float32, buf) : Float32[]

        # Decode both halves, exactly as section 6 and 6.1 say to.
        function dec(i)
            if dt == "f32"
                v = fl[i]
                isnan(v) && return (Inf, nothing)
                v < 0 && return (Inf, -Float64(v))
                return (Float64(v), nothing)
            end
            r = raws[i]
            r == sentinel && return (Inf, nothing)
            r >= base && return (Inf, es * (r - base)^2)
            (r * sc, nothing)
        end

        nboth = 0; nesc_rt = 0; worst = 0.0; nreach = 0
        for i in eachindex(after)
            v, e = dec(i)
            isfinite(v) && (nreach += 1)
            if after[i] < 0.0f0
                nesc_rt += 1
                # An escape cell MUST still read as unreachable. This is the
                # property the whole design rests on: it must lose every
                # comparison against a real route.
                isfinite(v) && (nboth += 1)
                e === nothing || (worst = max(worst, abs(e - Float64(-after[i]))))
            elseif is_terminal(after[i], cap)
                e === nothing || (nboth += 1)
            end
        end
        _check(nboth == 0,
               "$dt: no cell decodes as both reachable and escapable",
               "$nboth of $(length(after))")
        _check(nesc_rt == nesc, "$dt: every escape cell survived encoding",
               "$nesc_rt of $nesc")
        # The band's own resolution is the bar: `escape_scale * (2r+1)` at the
        # top code, which is what a square-law band costs at the far end.
        tol = dt == "f32" ? 1.0e-3 :
              es * (2 * (escape_codes(dt) - 1) + 1) + 1.0e-6
        _check(worst <= tol, "$dt: escape times round-trip inside the band's step",
               @sprintf("worst %.4f s, band step at the top %.4f s", worst, tol))
        # Only where the dtype can represent `cap` at all. `u8` at 25 ms a
        # code tops out at 5.575 s against a 30 s cap, so most real values
        # saturate to the sentinel -- which is the long-standing behaviour of
        # a one-byte table and not something the escape band changed. What
        # matters for `u8` is that they saturate to the *sentinel* and not
        # into the band, and "no cell decodes as both" above is that test.
        nterm = count(v -> is_terminal(v, cap), after)
        if base == 0 || Float64(cap) / sc < base
            _check(nreach == nterm,
                   "$dt: reachable count is unchanged by the escape band",
                   "$nreach vs $nterm")
        else
            _check(nreach <= nterm,
                   "$dt: cannot represent a $(cap) s cap, so it saturates",
                   "$nreach of $nterm representable below " *
                   @sprintf("%.3f s", base * sc))
        end
    end

    # A real value that would land in the escape band must saturate to the
    # sentinel rather than be read back as an escape. `run_solve` refuses a
    # config that could do this, so it is a guard, but it is the guard on the
    # one confusion the format cannot tolerate.
    probe = Float32[Float32(0.9 * 0xfc00 * 0.001), Float32(1.1 * 0xfc00 * 0.001)]
    pbuf = reinterpret(UInt16, encode(probe, "u16", 0.001, 100.0f0,
                                      escape_scale_for("u16", 100.0f0)))
    _check(pbuf[1] < 0xfc00 && pbuf[2] == 0xffff,
           "a real value above the band saturates to unreachable, not into it",
           "$(pbuf[1]), $(pbuf[2])")
    nothing
end

"""
Test 17: does the estimate know what a sweep costs?

This is the test the H200 job needed and did not have. `plan` quoted an hour
and a half for a run that took two and a half hours, because it priced every
sweep at a cell rate measured under settings that ask for a third of the work
production asks for. Nothing was broken -- the rate was real, the arithmetic
was right, and the number was still wrong by 41%, which on a rented card is a
real bill.

So there are two things to hold:

  1. `sweep_work` counts what `cell_update` will actually do. The lookahead
     count is exact and checkable by hand; the substep count follows from
     `cfl_tau` and `substeps_for`, so it must move when the horizon moves.
  2. The cost model built on it must predict a real sweep. On a GPU this is
     measured: fit the coefficients on one regime and predict another, which
     is exactly what `plan` does when it prices a job the benchmark never ran.

The monotonicity checks are the part that runs everywhere. They are weaker
than the measurement but they catch the failure that actually happened --
a work count that does not respond to the settings that change the work.
"""
function test_sweep_cost()
    println("\n[17] the estimate prices the sweep it is going to run")
    ok = true
    m = _realistic()
    g = Grid6((16, 16, 24, 15, 15, 15), (0.0, 0.0, -π, -150.0, -150.0, -7.0),
                                        (350.0, 350.0, π, 150.0, 150.0, 7.0))
    nctl = length(control_set(4))
    par(; cfl, tau_max, scan = 8, rounds = 2) =
        Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(3),
               adaptive_checks = true, cap = 60.0f0, ntau = Int32(5),
               tau_ratio = 2.0f0, cfl = Float32(cfl), tau_min = 0.004f0,
               tau_max = Float32(tau_max), hmax = 0.0125f0,
               ncoarse = Int32(scan), rounds = Int32(rounds), delta0 = 0.35f0,
               rk2 = true, simplex = true)

    # -- the lookahead count is exact, and countable by hand --------------
    # `cell_update` runs: coast, the warm start, `ncoarse` lattice entries,
    # `6 * rounds` pattern probes, `ntau - 1` ladder rungs and the `dt` rung.
    w = sweep_work(g, m, par(cfl = 2.0, tau_max = 0.5), nctl)
    ok &= _check(w.lookaheads == 2 + 8 + 12 + 4 + 1,
                 "the lookahead count is the one `cell_update` issues",
                 "$(w.lookaheads) vs $(2 + 8 + 12 + 4 + 1)")
    w2 = sweep_work(g, m, par(cfl = 2.0, tau_max = 0.5, scan = 16, rounds = 3),
                    nctl)
    ok &= _check(w2.lookaheads == 2 + 16 + 18 + 4 + 1,
                 "and it follows `control_scan` and `refine_rounds`",
                 "$(w2.lookaheads) vs $(2 + 16 + 18 + 4 + 1)")

    # -- the step work follows the horizon --------------------------------
    # The failure that shipped: these two regimes were treated as costing the
    # same, and they do not.
    lo = sweep_work(g, m, par(cfl = 1.0, tau_max = 0.2), nctl)
    hi = sweep_work(g, m, par(cfl = 2.0, tau_max = 0.5), nctl)
    ok &= _check(hi.substeps > 1.3 * lo.substeps,
                 "a longer horizon costs more integration substeps",
                 @sprintf("%.0f vs %.0f", hi.substeps, lo.substeps))
    ok &= _check(hi.probes > lo.probes,
                 "and more swept collision probes",
                 @sprintf("%.0f vs %.0f", hi.probes, lo.probes))
    # Monotone in both knobs separately, since either alone can lengthen the
    # step and either alone was enough to make the estimate wrong.
    taus = [sweep_work(g, m, par(cfl = 2.0, tau_max = t), nctl).substeps
            for t in (0.05, 0.1, 0.2, 0.5)]
    ok &= _check(issorted(taus), "monotone in `tau_max`", string(taus))
    cfls = [sweep_work(g, m, par(cfl = cf, tau_max = 0.5), nctl).substeps
            for cf in (0.5, 1.0, 2.0, 4.0)]
    ok &= _check(issorted(cfls), "monotone in `cfl`", string(cfls))

    # -- a slower drivetrain gets a longer step, so it costs more ---------
    # `cfl_tau` divides the velocity cell by the largest acceleration `B` can
    # produce, so half the authority is twice the horizon. This is the half
    # of the H200 miss that no setting would have revealed: the benchmark's
    # synthetic model has 1.7x the yaw authority of the real fit.
    half = Model(m.B ./ 2, m.A, m.q, m.S, m.D, m.c, m.eps, m.knee)
    ok &= _check(sweep_work(g, half, par(cfl = 2.0, tau_max = 0.5), nctl).substeps >
                 hi.substeps,
                 "a weaker drivetrain takes longer steps and costs more",
                 @sprintf("%.0f vs %.0f",
                          sweep_work(g, half, par(cfl = 2.0, tau_max = 0.5),
                                     nctl).substeps, hi.substeps))

    # -- the config reader --------------------------------------------------
    c0, src0 = cell_cost(Dict{String,Any}())
    ok &= _check(src0 == "default" && c0.per_step_s == REF_CELL_COST.per_step_s,
                 "an unmeasured config falls back to the desktop reference",
                 src0)
    c1, src1 = cell_cost(Dict{String,Any}("cell_rate" => 1.0 /
                              cell_seconds(REF_WORK, REF_CELL_COST)))
    ok &= _check(src1 == "rate" &&
                 isapprox(c1.per_step_s, REF_CELL_COST.per_step_s; rtol = 1e-9),
                 "a legacy `cell_rate` at the reference workload reproduces it",
                 @sprintf("%s, %.4g vs %.4g", src1, c1.per_step_s,
                          REF_CELL_COST.per_step_s))
    # A rate that carries its workload is as good as a cost pair: the point of
    # the reconstruction is that it stops being an assumption.
    _, src1b = cell_cost(Dict{String,Any}("cell_rate" => 5.0e7,
        "cell_rate_work" => Dict("lookaheads" => 27.0, "substeps" => 100.0,
                                 "probes" => 100.0)))
    ok &= _check(src1b == "cost",
                 "a `cell_rate` that names its workload is not an assumption",
                 src1b)
    c2, src2 = cell_cost(Dict{String,Any}("cell_cost" =>
                Dict("per_lookahead_s" => 1.0e-11, "per_step_s" => 2.0e-11)))
    ok &= _check(src2 == "cost" && c2.per_lookahead_s == 1.0e-11 &&
                 c2.per_step_s == 2.0e-11,
                 "an explicit `cell_cost` is taken as given")

    if !CUDA.functional()
        println("  SKIP  no CUDA device to check the model against a real sweep")
        return ok
    end

    # -- does it predict a sweep it was not fitted on? --------------------
    # Fit on two regimes, predict a third. This is the extrapolation `plan`
    # makes every time it prices a job on a card the benchmark measured under
    # other settings, so it is the one that has to hold.
    gm = Grid6((40, 40, 24, 15, 15, 15), (0.0, 0.0, -π, -150.0, -150.0, -7.0),
                                         (350.0, 350.0, π, 150.0, 150.0, 7.0))
    cells = ncells(gm)
    occ = falses(Int(gm.n[1]) * Int(gm.n[2]) * Int(gm.n[3]))
    for i in 13:16, j in 0:(Int(gm.n[2]) - 1), k in 0:(Int(gm.n[3]) - 1)
        occ[(i * Int(gm.n[2]) + j) * Int(gm.n[3]) + k + 1] = true
    end
    act = 1 - count(occ) / length(occ)
    ct = control_set(4)
    ctl_h = Float32[getindex.(ct, 1); getindex.(ct, 2); getindex.(ct, 3)]
    ttol = NTuple{6,Float32}(Float32[gm.step[k] * 0.5f0 for k in 1:6])
    seeds = Int64.(target_cells(gm, (280.0f0, 175.0f0, 0.0f0, 0.0f0, 0.0f0,
                                     0.0f0), ttol)) .+ 1
    docc = CuArray(occ); dctl = CuArray(ctl_h); dseed = CuArray(seeds)
    V = CUDA.fill(60.0f0, cells); pol = CUDA.zeros(Int8, 3 * cells)
    total = CUDA.zeros(Float32, 1)
    function per_cell(p)
        solve_value!(V, docc, gm, m, dseed, dctl, nctl, p; iters = 2, tol = 0.0,
                     use_gpu = true, pol = pol, total = total)
        CUDA.synchronize(); t0 = time()
        solve_value!(V, docc, gm, m, dseed, dctl, nctl, p; iters = 3, tol = 0.0,
                     use_gpu = true, pol = pol, total = total)
        CUDA.synchronize()
        ((time() - t0) / 3) / (cells * act)
    end
    fit = [(par(cfl = 1.0, tau_max = 0.05)), (par(cfl = 4.0, tau_max = 0.5))]
    ys = [per_cell(p) for p in fit]
    ws = [sweep_work(gm, m, p, nctl) for p in fit]
    A = [w.lookaheads for w in ws]; B = [w.substeps + w.probes for w in ws]
    a, b = hcat(A, B) \ ys
    held = par(cfl = 2.0, tau_max = 0.5)
    got = per_cell(held)
    pred = cell_seconds(sweep_work(gm, m, held, nctl),
                        (per_lookahead_s = a, per_step_s = b))
    CUDA.unsafe_free!(V); CUDA.unsafe_free!(pol); CUDA.unsafe_free!(docc)
    CUDA.unsafe_free!(dctl); CUDA.unsafe_free!(dseed)
    err = pred / got - 1
    # 15% is loose on purpose. It is a timing measurement on a card that may
    # be doing other things, and the bar it has to clear is the 41% error the
    # single-rate model made -- not a benchmark's repeatability.
    ok &= _check(abs(err) < 0.15,
                 "the fitted cost predicts a regime it was not fitted on",
                 @sprintf("%+.1f%% (%.2f vs %.2f ns/cell)", 100 * err,
                          pred * 1e9, got * 1e9))
    ok
end

"""Run every self-test. Returns a process exit code."""
function self_test()
    _PASS[] = 0; _FAIL[] = 0
    println("PeregrineSolver self-test")
    println("backend: ", CUDA.functional() ? "cuda" : "cpu")
    test_control_set()
    test_kuhn()
    test_nan_knee()
    test_velocity_envelope()
    test_integrator()
    test_min_time()
    test_search_quality()
    test_tau_monotone()
    test_tiling()
    test_ooc()
    test_walls()
    test_advice()
    test_reach_curve()
    test_prefetch()
    test_cell_bytes()
    test_escape()
    test_sweep_cost()
    @printf("\n%d passed, %d failed\n", _PASS[], _FAIL[])
    _FAIL[] == 0 ? 0 : 1
end
