#!/usr/bin/env julia
#
# Compare solver schemes on speed AND on the quality of the policy they
# produce.
#
#   julia --project=solver solver/bench/compare.jl <config.json> [--grid a,b,c,d,e,f]
#         [--targets 1] [--starts 150] [--schemes name,name]
#
# Schemes are declared in `SCHEMES` below as overrides on the config. Every
# scheme solves the same grid, the same occupancy and the same seed cells, so
# the only thing that differs is the Bellman update -- and every resulting
# table is scored by the same rollout controller.

include(joinpath(@__DIR__, "harness.jl"))

using CUDA

# --------------------------------------------------------------------------

function parse_args(args)
    o = Dict{String,Any}("grid" => nothing, "targets" => 1, "starts" => 150,
                         "schemes" => nothing, "iters" => nothing)
    i = 2
    while i <= length(args)
        a = args[i]
        if a == "--grid"
            o["grid"] = Tuple(parse.(Int, split(args[i+1], ","))); i += 2
        elseif a == "--targets"
            o["targets"] = parse(Int, args[i+1]); i += 2
        elseif a == "--starts"
            o["starts"] = parse(Int, args[i+1]); i += 2
        elseif a == "--iters"
            o["iters"] = parse(Int, args[i+1]); i += 2
        elseif a == "--schemes"
            o["schemes"] = String.(split(args[i+1], ",")); i += 2
        else
            error("unknown option '$a'")
        end
    end
    o
end

"""
Solver settings under test. Keys override the config file's own.

`base_*` reproduce the original scheme -- whole lattice every sweep, no
refinement, one step length, fixed collision probes -- so they are the
reference the optimised schemes have to beat on BOTH axes.
"""
const OFF = Dict{String,Any}("control_scan" => 0, "refine_rounds" => 0,
                             "tau_levels" => 1, "warm_start" => false,
                             "adaptive_checks" => false, "rk2" => false, "cfl" => 0.0,
                             "simplex" => false)

base(level) = merge(OFF, Dict{String,Any}("control_level" => level))

const SCHEMES = Dict{String,Dict{String,Any}}(
    "base_l3" => base(3),
    "base_l4" => base(4),
    "base_l6" => base(6),

    # One change at a time, so a regression can be attributed.
    "warm"    => merge(OFF, Dict{String,Any}("control_level" => 4,
                    "warm_start" => true, "control_scan" => 8)),
    "refine"  => merge(OFF, Dict{String,Any}("control_level" => 4,
                    "warm_start" => true, "control_scan" => 8,
                    "refine_rounds" => 2)),
    "achecks" => merge(OFF, Dict{String,Any}("control_level" => 4,
                    "warm_start" => true, "control_scan" => 8,
                    "refine_rounds" => 2, "adaptive_checks" => true)),
    "rk2"     => merge(OFF, Dict{String,Any}("control_level" => 4,
                    "warm_start" => true, "control_scan" => 8,
                    "refine_rounds" => 2, "adaptive_checks" => true,
                    "rk2" => true)),
    # Several horizons, but still anchored to the configured dt.
    "tau"     => merge(OFF, Dict{String,Any}("control_level" => 4,
                    "warm_start" => true, "control_scan" => 8,
                    "refine_rounds" => 2, "adaptive_checks" => true,
                    "rk2" => true, "tau_levels" => 4)),
    # Horizons derived from the grid instead. This is the change that answers
    # "let the solver pick the step length".
    "cfl"     => merge(OFF, Dict{String,Any}("control_level" => 4,
                    "warm_start" => true, "control_scan" => 8,
                    "refine_rounds" => 2, "adaptive_checks" => true,
                    "rk2" => true, "tau_levels" => 4, "cfl" => 1.0)),

    # What the longest lookahead is worth.
    #
    # `tau_max` is the lever the out-of-core driver hangs on: it caps the top
    # of the step-length ladder, which is exactly the dependency reach, which
    # is the halo every tile is loaded with. Lowering it is what makes a
    # full-scale grid tile efficiently, so what it costs in accuracy is a
    # number worth having rather than a shrug. Everything else here is the
    # production scheme, so the only difference is the cap.
    #
    # Run this on a grid whose CELLS are the size the real run will use --
    # the ladder is clipped when `tau_max` bites below `8*tau0`, and `tau0`
    # is derived from the cell -- or the answer will not transfer.
    "tmax_050" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0,
                    "tau_max" => 0.5),
    "tmax_020" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0,
                    "tau_max" => 0.2),
    "tmax_010" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0,
                    "tau_max" => 0.1),
    "tmax_006" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0,
                    "tau_max" => 0.06),

    # Cheaper and richer variants of the full scheme.
    "opt_fast" => Dict{String,Any}("control_level" => 3, "control_scan" => 4,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 3, "adaptive_checks" => true, "cfl" => 1.0),
    "opt"      => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 4, "adaptive_checks" => true, "cfl" => 1.0),
    "opt_rich" => Dict{String,Any}("control_level" => 4, "control_scan" => 12,
                    "refine_rounds" => 3, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 1.0),
    # The ladder's reach, not its base, turned out to be what matters: a
    # 4-rung ladder anchored at dt = 0.05 beat one anchored at a smaller
    # grid-derived tau0, purely because its top rung was longer. These two
    # test that directly -- more rungs, and a wider base.
    "opt_tau5" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 1.0),
    "opt_cfl2" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0),

    # Cost is very nearly linear in the number of candidates evaluated, and
    # the refinement is the expensive part: 12 of 22 candidates for a few
    # percent, against 3 candidates for thirty. But the incumbent persists in
    # `pol` between sweeps, so refinement ACCUMULATES -- one round a sweep
    # over three hundred sweeps should reach the same place as two. These
    # test whether the second round is buying anything.
    "opt_r1"  => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 1, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0),
    "opt_lean" => Dict{String,Any}("control_level" => 4, "control_scan" => 4,
                    "refine_rounds" => 1, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true, "cfl" => 2.0),

    # It is the ladder's REACH that pays, not its base. Widening the base
    # (cfl 1 -> 2) beat adding a shorter bottom rung, and did it while
    # converging in far fewer sweeps -- long steps carry information across
    # the grid faster. This tests whether the 0.5 s cap is now the binding
    # constraint rather than the ladder.
    "opt_far" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true,
                    "cfl" => 2.0, "tau_max" => 1.0),

    # The interpolant. `opt_ml` is the full scheme still reading 64 corners;
    # `opt_kuhn` is the same scheme on 7-point simplex interpolation, which is
    # both cheaper and what the robot actually uses to read the finished
    # table. Comparing these two isolates the interpolant from everything
    # else that changed.
    "opt_ml"   => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true,
                    "cfl" => 2.0, "simplex" => false),
    "opt_kuhn" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true,
                    "cfl" => 2.0, "simplex" => true),

    # Is the collision check keeping up with the long horizons?
    #
    # The optimised schemes hit obstacles ~2.5x more often in rollout than the
    # baseline. Two very different explanations fit that: either they simply
    # move (the baseline times out on two thirds of starts, and a robot that
    # does not go anywhere does not hit anything), or a long step is TUNNELLING
    # through an obstacle because the swept probe count is too coarse -- which
    # would lower V and drive the robot into walls, and would be a soundness
    # bug rather than a behaviour difference.
    #
    # These separate the two. If tunnelling is happening, forcing many more
    # probes must RAISE meanV and cut the obstacle count. If the numbers barely
    # move, the checking was already adequate and the obstacle rate is the
    # honest cost of actually driving.
    "opt_chk12" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true,
                    "cfl" => 2.0, "simplex" => true, "sweep_checks" => 12),
    "opt_chk24" => Dict{String,Any}("control_level" => 4, "control_scan" => 8,
                    "refine_rounds" => 2, "warm_start" => true,
                    "tau_levels" => 5, "adaptive_checks" => true,
                    "cfl" => 2.0, "simplex" => true, "sweep_checks" => 24),
)

const ORDER = ["base_l4", "opt"]

# --------------------------------------------------------------------------

function solve_scheme(case::Case, ti::Int, over::Dict{String,Any}, opts_cli)
    cfg = case.cfg
    g, m, occ = case.g, case.m, case.occ
    gv(k, d) = haskey(over, k) ? over[k] : PS.getc(cfg, k, d)

    level  = Int(gv("control_level", 3))
    dt     = Float32(gv("dt", 0.05))
    nsub   = Int(gv("substeps", 4))
    checks = Int(gv("sweep_checks", 3))
    iters  = opts_cli["iters"] === nothing ? Int(gv("iterations", 400)) :
             opts_cli["iters"]
    tol    = Float64(gv("tolerance", 1e-3))
    cap    = Float32(gv("value_cap", 60.0))
    nearest = Bool(gv("nearest", false))

    ctl_t = PS.control_set(level)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]

    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    tcells = PS.target_cells(g, case.states[ti], ttol)

    p = PS.Params(dt = dt, nsub = Int32(nsub), checks = Int32(checks),
                  adaptive_checks = Bool(gv("adaptive_checks", true)),
                  nearest = nearest, cap = cap,
                  ntau = Int32(gv("tau_levels", 4)),
                  tau_ratio = Float32(gv("tau_ratio", 2.0)),
                  cfl = Float32(gv("cfl", 1.0)),
                  tau_min = Float32(gv("tau_min", 0.004)),
                  tau_max = Float32(gv("tau_max", 0.5)),
                  hmax = Float32(gv("substep_max", dt / max(nsub, 1))),
                  ncoarse = Int32(gv("control_scan", 8)),
                  rounds = Int32(gv("refine_rounds", 2)),
                  delta0 = Float32(gv("refine_delta", 0.35)),
                  rk2 = Bool(gv("rk2", true)),
                  vclamp = Bool(gv("velocity_clamp", false)),
                  simplex = Bool(gv("simplex", true)))

    ncell = PS.ncells(g)
    V = CUDA.fill(cap, ncell)
    occ_d = CuArray(occ)
    ctl_d = CuArray(ctl_h)
    total = CUDA.zeros(Float32, 1)
    tidx = CuArray(Int64.(tcells) .+ 1)
    pol = Bool(gv("warm_start", true)) ? CUDA.zeros(Float32, 3 * ncell) : nothing

    # One untimed sweep first: the first launch pays JIT compilation, which
    # would otherwise land entirely on whichever scheme happens to run first.
    PS.sweep_gpu!(V, occ_d, pol, g, m, ctl_d, nctl, p, 0, total)

    CUDA.synchronize()
    t0 = time()
    its, delta = PS.solve_value!(V, occ_d, g, m, tidx, ctl_d, nctl, p;
                                 iters = iters, tol = tol, use_gpu = true,
                                 pol = pol, total = total)
    CUDA.synchronize()
    el = time() - t0

    Vh = Array(V)
    CUDA.unsafe_free!(V); CUDA.unsafe_free!(occ_d); CUDA.unsafe_free!(ctl_d)
    CUDA.unsafe_free!(total); CUDA.unsafe_free!(tidx)
    pol === nothing || CUDA.unsafe_free!(pol)

    (V = Vh, iters = its, delta = delta, seconds = el, sweep_s = el / its,
     nctl = nctl, reached = count(v -> v < cap, Vh) / length(Vh), ttol = ttol)
end

function main()
    isempty(ARGS) && (println(stderr, "usage: compare.jl <config.json> [...]"); return 2)
    o = parse_args(ARGS)
    case = load_case(ARGS[1]; grid_override = o["grid"])
    @printf("grid %s = %d cells   obstacles blocked %.1f%%\n",
            string(Int.(case.g.n)), PS.ncells(case.g),
            100 * count(case.occ) / length(case.occ))

    starts = sample_starts(case.g, case.occ, o["starts"])
    # A rich rollout control set and a short rollout step, so the rollout is
    # limited by the table rather than by its own controller.
    rctl = PS.control_set(6)
    println("rollout: robot rule max dot(-grad V, f), $(length(rctl)) controls, " *
            "0.02 s step, $(length(starts)) starts, handoff 15 cm / 0.25 rad / 40 cm-s")

    names = o["schemes"] === nothing ? ORDER : o["schemes"]
    cap = Float32(PS.getc(case.cfg, "value_cap", 60.0))

    for ti in 1:o["targets"]
        println("=== target $(ti-1): $(case.names[ti]) ===")
        rs = Dict{String,Any}()
        for nm in names
            haskey(SCHEMES, nm) || error("no scheme '$nm'")
            r = solve_scheme(case, ti, SCHEMES[nm], o)
            rs[nm] = r
            @printf("%-10s ctl %3d  %4d sw  %7.2fs (%.3f s/sw)  reach %.4f\n",
                    nm, r.nctl, r.iters, r.seconds, r.sweep_s, r.reached)
            flush(stdout)
        end

        # Mean value over the cells EVERY scheme reached.
        #
        # This is the sharpest accuracy signal available, and it needs no
        # reference table. Every candidate control a scheme tries is a real
        # Bellman evaluation, never an approximation of one, so every scheme's
        # V is an upper bound on the same true value function. Two valid upper
        # bounds compare directly: the lower one is the tighter one, and
        # therefore the more accurate. It is also far less noisy than rollout,
        # because it averages over every cell instead of a sample of starts.
        #
        # Restricting to the common reached set stops a scheme from looking
        # good merely by leaving the expensive cells unreached.
        mask = trues(length(rs[names[1]].V))
        for nm in names
            mask .&= (rs[nm].V .< cap - 1.0f-3)
        end
        nmask = count(mask)
        base_mean = sum(@view rs[names[1]].V[mask]) / max(nmask, 1)
        @printf("\ncommon reached cells: %d (%.3f)\n", nmask, nmask / length(mask))

        for nm in names
            V = rs[nm].V
            mv = sum(@view V[mask]) / max(nmask, 1)
            lower = count(i -> V[i] < rs[names[1]].V[i] - 1.0f-4, findall(mask))
            higher = count(i -> V[i] > rs[names[1]].V[i] + 1.0f-4, findall(mask))
            @printf("%-10s meanV %8.4f s  (%+7.3f%% vs %s)  tighter %.3f / looser %.3f\n",
                    nm, mv, 100 * (mv - base_mean) / base_mean, names[1],
                    lower / max(nmask, 1), higher / max(nmask, 1))
        end

        # Rollout: the reality check. Mean V can only be trusted while every
        # scheme's V really is a valid upper bound, and a policy that actually
        # drives the robot there is what proves it.
        println()
        ss = Dict{String,Any}()
        for nm in names
            r = rs[nm]
            s = score(r.V, case.g, case.m, case.occ, case.states[ti],
                      starts, rctl; cap = cap)
            ss[nm] = s
            @printf("%-10s arrive %.3f  censored %6.2f s  median %5.2f  mean %5.2f   failed: %s\n",
                    nm, s.arrived, s.censored, s.median_time, s.mean_time,
                    why_str(s.why))
            flush(stdout)
        end
        for nm in names[2:end]
            c = compare(ss[names[1]].times, ss[nm].times)
            @printf("  %-10s vs %-10s  n=%3d  median %.2f -> %.2f s  mean %.2f -> %.2f s  better %d / worse %d / tie %d\n",
                    nm, names[1], c.n, c.med_a, c.med_b, c.mean_a, c.mean_b,
                    c.wins_b, c.losses_b, c.ties)
        end
        println()
    end
    return 0
end

exit(main())
