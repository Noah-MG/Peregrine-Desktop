#!/usr/bin/env julia
#
# Why does a rollout fail?
#
# Solves one target, then prints the value at each start state and traces a
# single rollout step by step. A rollout harness that quietly never arrives
# would make every scheme look identical, so this exists to prove the harness
# is measuring the policy rather than its own bugs.

include(joinpath(@__DIR__, "harness.jl"))
using CUDA

function main()
    cfgpath = ARGS[1]
    gridarg = length(ARGS) >= 2 ? Tuple(parse.(Int, split(ARGS[2], ","))) : nothing
    case = load_case(cfgpath; grid_override = gridarg)
    g, m, occ = case.g, case.m, case.occ
    cap = 60.0f0

    ctl_t = PS.control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    p = PS.Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(3),
                  adaptive_checks = false, nearest = false, cap = cap,
                  ntau = Int32(1), ncoarse = Int32(0), rounds = Int32(0),
                  rk2 = false, vclamp = false)

    tcells = PS.target_cells(g, case.states[1], ttol)
    V = CUDA.fill(cap, PS.ncells(g))
    occ_d = CuArray(occ); ctl_d = CuArray(ctl_h)
    total = CUDA.zeros(Float32, 1); tidx = CuArray(Int64.(tcells) .+ 1)
    its, d = PS.solve_value!(V, occ_d, g, m, tidx, ctl_d, nctl, p;
                             iters = 400, tol = 1e-3, use_gpu = true, total = total)
    Vh = Array(V)
    @printf("solved in %d sweeps, delta %.3g, reached %.3f, seed cells %d\n",
            its, d, count(v -> v < cap, Vh) / length(Vh), length(tcells))

    targ = case.states[1]
    @printf("target %s  tol %s\n", string(targ), string(ttol))

    starts = sample_starts(g, occ, 40)
    vs = [PS.interp(Vh, g, s..., false, cap) for s in starts]
    fin = filter(v -> v < cap - 1.0f-3, vs)
    @printf("V at starts: %d/%d below cap, min %.2f median %.2f max %.2f\n",
            length(fin), length(vs),
            isempty(fin) ? NaN : minimum(fin),
            isempty(fin) ? NaN : sort(fin)[cld(length(fin), 2)],
            isempty(fin) ? NaN : maximum(fin))

    # Trace the first start whose value is finite.
    k = findfirst(v -> v < cap - 1.0f-3, vs)
    k === nothing && (println("no start has a finite value"); return 0)
    s0 = starts[k]
    @printf("\ntrace from %s  V = %.3f\n", string(s0), vs[k])

    rctl = PS.control_set(4)
    x, y, h, vfx, vfy, w = s0
    dt_r = 0.02f0
    for step in 0:600
        v_now = PS.interp(Vh, g, x, y, h, vfx, vfy, w, false, cap)
        if step % 25 == 0
            @printf("t=%5.2f  V=%7.3f  pos=(%7.2f,%7.2f) h=%6.3f  v=(%7.2f,%7.2f) w=%6.2f\n",
                    step * dt_r, v_now, x, y, h, vfx, vfy, w)
        end
        dh = h - targ[3]; dh -= 2f0 * Float32(pi) * round(dh / (2f0 * Float32(pi)))
        if abs(x - targ[1]) <= ttol[1] && abs(y - targ[2]) <= ttol[2] &&
           abs(dh) <= ttol[3] && abs(vfx - targ[4]) <= ttol[4] &&
           abs(vfy - targ[5]) <= ttol[5] && abs(w - targ[6]) <= ttol[6]
            @printf("ARRIVED at t = %.2f\n", step * dt_r); return 0
        end
        sh, ch = sincos(h)
        vx = ch * vfx + sh * vfy; vy = -sh * vfx + ch * vfy
        best = Float32(Inf); bu = (0f0, 0f0, 0f0)
        for u in rctl
            nx, ny, nh, nvx, nvy, nw =
                PS.step_state(m, x, y, h, vx, vy, w, u[1], u[2], u[3], dt_r, Int32(4))
            snh, cnh = sincos(nh)
            q = PS.interp(Vh, g, nx, ny, nh, cnh*nvx - snh*nvy, snh*nvx + cnh*nvy,
                          nw, false, cap)
            c = dt_r + q
            c < best && (best = c; bu = u)
        end
        x, y, h, vx, vy, w =
            PS.step_state(m, x, y, h, vx, vy, w, bu[1], bu[2], bu[3], dt_r, Int32(4))
        sh, ch = sincos(h)
        vfx = ch * vx - sh * vy; vfy = sh * vx + ch * vy
    end
    println("did not arrive in 12 s")
    0
end

exit(main())
