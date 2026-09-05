#!/usr/bin/env julia
#
# Measure what a machine is actually worth to this solver -- what a cell
# update costs it and what its disk does -- plus what the prefetch buys.
#
#   julia --project=solver -t auto solver/cloud/benchmark.jl [--seconds 30]
#
# Prints a JSON blob on the last line. Everything before it is commentary.
#
# Why this exists: `plan` costs a run from `cell_cost` and `disk_rate`, and
# both defaults are measurements of one 8 GB desktop card and one SATA SSD.
# On any other machine the estimate is an extrapolation, and the whole point
# of the estimate is to decide how long to rent for. Ten minutes here
# replaces the guess with a number.
#
# It needs no field, no targets and no regression: the model is the same
# synthetic fit the self-test uses, so the answer is a property of the
# machine rather than of a particular workspace. That last part is why the
# answer is a cost *pair* rather than a cell rate -- see `measure_cell_cost`.
# A rate is a property of a machine AND a workload, and quoting one measured
# here against a workload solved there is what made an H200 job that took two
# and a half hours get planned as an hour and a half.

include(joinpath(@__DIR__, "..", "src", "PeregrineSolver.jl"))
using .PeregrineSolver
using .PeregrineSolver: Grid6, Params, Model, ncells, control_set, solve_value!,
                        solve_value_ooc!, target_cells, reach_extent,
                        halo_cells, tile_plan, open_store, open_policy_store,
                        close_store!, _realistic, cell_bytes, sweep_work,
                        effective_rate, REF_CELL_COST
using CUDA, JSON3, Printf

const WANT_S = let i = findfirst(==("--seconds"), ARGS)
    i === nothing ? 25.0 : parse(Float64, ARGS[i + 1])
end

say(s) = (println("\n\033[1;36m== ", s, "\033[0m"); flush(stdout))
note(s) = (println("   ", s); flush(stdout))

# --------------------------------------------------------------------------

"""A grid of about `want` cells, in the proportions a real run uses."""
function sized_grid(want::Float64)
    # Hold the short axes fixed at production-ish counts and move x and y,
    # which is what a real resolution change mostly does.
    nh, nv, nw = 24, 15, 15
    col = nh * nv * nv * nw
    s = max(8, round(Int, sqrt(want / col)))
    n = (s, s, nh, nv, nv, nw)
    Grid6(n, (0.0, 0.0, -π, -150.0, -150.0, -7.0),
             (350.0, 350.0, π, 150.0, 150.0, 7.0))
end

"""
The solver parameters for one measurement point.

The defaults are production's -- `cfl` 2, `tau_max` 0.5, three collision
probes -- and NOT the short-step settings this file used to measure at. That
was the bug the cost model exists to fix: a rate measured at `cfl` 1 /
`tau_max` 0.2 and spent on a `cfl` 2 / `tau_max` 0.5 run under-quoted a real
H200 job by 41%, because those settings buy a longer horizon and a longer
horizon buys more integration substeps per lookahead.
"""
bench_params(; cfl = 2.0, tau_max = 0.5, checks = 3) =
    Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(checks),
           adaptive_checks = true, cap = 60.0f0, ntau = Int32(5),
           tau_ratio = 2.0f0, cfl = Float32(cfl), tau_min = 0.004f0,
           tau_max = Float32(tau_max), hmax = 0.0125f0, ncoarse = Int32(8),
           rounds = Int32(2), delta0 = 0.35f0, rk2 = true, simplex = true)

function make_case(g::Grid6; p::Params = bench_params())
    m = _realistic()
    n = g.n
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    # One slab, so the swept collision probe is exercised rather than skipped.
    lo = max(1, Int(n[1]) ÷ 3); hi = min(Int(n[1]) - 1, lo + 3)
    for i in lo:hi, j in 0:(Int(n[2]) - 1), k in 0:(Int(n[3]) - 1)
        occ[(i * Int(n[2]) + j) * Int(n[3]) + k + 1] = true
    end
    ctl_t = control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]
    targ = (280.0f0, 175.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    seeds = Int64.(target_cells(g, targ, ttol)) .+ 1
    (m = m, occ = Vector{Bool}(occ), p = p, nctl = nctl, ctl_h = ctl_h,
     seeds = seeds)
end

"""
Seconds a sweep of `g` takes on this card under `p`.

`warmups` sweeps are thrown away first: they pay for compilation, for the
first touch of every page, and for giving the warm-start policy something to
be warm about -- a cell whose incumbent is still coast skips one of the
lookaheads the cost model counts. `min_iters` bounds the timed run, which is
what keeps a full-size grid from costing minutes for a number that needs
seconds.
"""
function time_sweep(g::Grid6, p::Params, want_s::Float64;
                    warmups::Int = 2, min_iters::Int = 3)
    cells = ncells(g)
    c = make_case(g; p = p)
    V = CUDA.fill(p.cap, cells)
    pol = CUDA.zeros(Int8, 3 * cells)   # the layout production allocates
    docc = CuArray(c.occ); dctl = CuArray(c.ctl_h)
    dseed = CuArray(c.seeds); total = CUDA.zeros(Float32, 1)
    solve_value!(V, docc, g, c.m, dseed, dctl, c.nctl, p; iters = warmups,
                 tol = 0.0, use_gpu = true, pol = pol, total = total)
    CUDA.synchronize()
    t1 = time()
    solve_value!(V, docc, g, c.m, dseed, dctl, c.nctl, p; iters = 1,
                 tol = 0.0, use_gpu = true, pol = pol, total = total)
    CUDA.synchronize(); one_s = time() - t1
    iters = clamp(round(Int, want_s / max(one_s, 1e-6)), min_iters, 200)
    t0 = time()
    solve_value!(V, docc, g, c.m, dseed, dctl, c.nctl, p; iters = iters,
                 tol = 0.0, use_gpu = true, pol = pol, total = total)
    CUDA.synchronize()
    el = time() - t0
    CUDA.unsafe_free!(V); CUDA.unsafe_free!(pol)
    CUDA.unsafe_free!(docc); CUDA.unsafe_free!(dctl); CUDA.unsafe_free!(dseed)
    blocked = count(c.occ) / length(c.occ)
    (sweep_s = el / iters, sweeps = iters, seconds = el, cells = cells,
     active = 1.0 - blocked,
     work = sweep_work(g, c.m, p, c.nctl))
end

"""
What a cell update costs here, as the two coefficients `plan` prices with.

**Not one rate.** A cell update is a fixed number of lookaheads, but each one
costs `substeps_for` RK2 substeps and a swept probe count that both come from
the horizon `cfl_tau` derives from the grid and the model -- so the same card
sweeping the same cells runs at very different rates depending on settings
this file cannot know. Measuring one point and calling it "the" cell rate is
what made a two-and-a-half hour H200 solve look like an hour and a half.

So: sweep the same grid under several horizon regimes, and least-squares fit

    seconds per active cell = a * lookaheads + b * (substeps + probes)

against what `sweep_work` says each regime asks for. Two coefficients from
five points; the residuals are reported so a bad fit is visible rather than
silently shipped. `plan` then prices any grid, model and parameter set --
including ones no benchmark regime resembled.

The regimes are chosen to spread the per-cell cost about 3x, which is roughly
the spread real configs cover, and to bracket production rather than sit at
one end of it.
"""
function measure_cell_cost(free::Int64)
    # A quarter of what is free, at the 16 bytes an in-core cell costs (V plus
    # the Float32 warm-start policy). A quarter rather than all of it because
    # this is a rate measurement, not a capacity test, and a grid that only
    # just fits would measure the allocator.
    want = min(free * 0.25 / cell_bytes(true), 6.0e8)
    g = sized_grid(want)
    note(@sprintf("grid %s = %d cells (%.2f GB in core)",
                  string(Int.(g.n)), ncells(g), ncells(g) * cell_bytes(true) / 2^30))

    regimes = [("production   cfl 2   tau_max 0.5 ", bench_params()),
               ("short step   cfl 1   tau_max 0.2 ", bench_params(cfl = 1.0, tau_max = 0.2)),
               ("shortest     cfl 1   tau_max 0.05", bench_params(cfl = 1.0, tau_max = 0.05)),
               ("long step    cfl 4   tau_max 0.5 ", bench_params(cfl = 4.0, tau_max = 0.5)),
               ("mid          cfl 2   tau_max 0.15", bench_params(cfl = 2.0, tau_max = 0.15))]
    # Split the time budget across the regimes rather than spending it all on
    # one, so the whole fit costs what the single measurement used to.
    each = max(WANT_S / length(regimes), 3.0)
    pts = NamedTuple[]
    for (name, p) in regimes
        r = time_sweep(g, p, each)
        ns = 1.0e9 * r.sweep_s / (r.cells * r.active)
        note(@sprintf("%s  %6.3f s/sweep   %6.2f ns/cell   %5.1f substeps %5.1f probes",
                      name, r.sweep_s, ns, r.work.substeps, r.work.probes))
        push!(pts, (name = strip(name), p = p, r = r, ns = ns))
    end

    # Least squares on the two columns. Both are per *active* cell: a blocked
    # cell returns before its first lookahead, and the benchmark's slab is a
    # twentieth of the grid, so leaving it in would bias both coefficients low.
    A = [pt.r.work.lookaheads for pt in pts]
    B = [pt.r.work.substeps + pt.r.work.probes for pt in pts]
    y = [pt.r.sweep_s / (pt.r.cells * pt.r.active) for pt in pts]
    a, b = hcat(A, B) \ y
    resid = [(A[i] * a + B[i] * b) / y[i] - 1 for i in eachindex(y)]
    worst = maximum(abs, resid)
    # A negative coefficient means the fit has gone through the points rather
    # than along them -- too little spread, or a noisy box. The model is still
    # usable with the fixed term dropped, and saying so beats shipping a
    # coefficient that makes a bigger grid cost less.
    if a < 0.0 || b <= 0.0
        b = sum(y) / sum(B); a = 0.0
        note("\033[1;33mfit degenerate -- falling back to a single per-step rate\033[0m")
    end
    note(@sprintf("shape: %.4f ns/lookahead + %.4f ns/step-unit  (worst residual %+.1f%%)",
                  a * 1e9, b * 1e9, 100 * worst))
    if worst > 0.15
        note("\033[1;33mthat is a loose fit; estimates from it are rough\033[0m")
    end

    # Second stage: one sweep of a grid the size people actually solve.
    #
    # The regimes above are swept on a quarter of free VRAM, which is small
    # and fast and gives the fit the SHAPE it needs -- how cost moves with
    # the horizon. It does not give the level. A cell update reads its
    # successor's value through a 7-point simplex interpolation at a scattered
    # address, so the cost per unit of work climbs as the value function
    # outgrows the caches: measured +14% between a 73M-cell grid and a
    # 731M-cell one on an 8 GB card, still climbing at the top. Extrapolating
    # a quarter-VRAM measurement to a grid four times the size is the second
    # reason the H200 estimate came in low, after the workload itself.
    #
    # So: predict this big sweep from the shape, measure it, and scale both
    # coefficients by the ratio. One sweep, at the settings production uses.
    # Sized to the biggest grid this card can hold, capped near the biggest
    # anyone solves. The cap matters on a large card: two thirds of an H200
    # is fourteen billion cells, an order of magnitude past any table that
    # fits the 8 GB size budget, and calibrating against a working set nobody
    # will ever have would trade one extrapolation for another.
    scale = 1.0
    big = try
        gb = sized_grid(min(free * 0.7 / cell_bytes(true), 2.0e9))
        ncells(gb) > 1.3 * ncells(g) ?
            time_sweep(gb, bench_params(), 0.0; warmups = 1, min_iters = 2) :
            nothing
    catch e
        note("full-size check skipped: " * first(sprint(showerror, e), 120))
        nothing
    end
    if big !== nothing
        want_s = big.sweep_s
        pred = (a * big.work.lookaheads +
                b * (big.work.substeps + big.work.probes)) *
               big.cells * big.active
        scale = want_s / max(pred, 1.0e-9)
        note(@sprintf("full size: %d cells, %.3f s/sweep against %.3f s predicted -> x%.3f",
                      big.cells, want_s, pred, scale))
        # A big correction means the two stages disagree about the machine,
        # not about the workload -- most often another process on the card.
        if !(0.5 < scale < 2.0)
            note("\033[1;33mthat is a large disagreement; is the card shared?\033[0m")
            scale = clamp(scale, 0.5, 2.0)
        end
        a *= scale; b *= scale
    end

    cost = (per_lookahead_s = a, per_step_s = b)
    note(@sprintf("\033[1;32m%.4f ns/lookahead + %.4f ns/step-unit\033[0m",
                  a * 1e9, b * 1e9))

    # The scalar the old config called `cell_rate`, quoted at the production
    # regime so it is at least the right order for the runs people do. Kept
    # for continuity, and for anything still reading it.
    prod = pts[1]
    rate = prod.r.cells * prod.r.active / prod.r.sweep_s
    # What the desktop card this project was tuned on would do on THIS case,
    # so "2.4x faster" is a comparison of two machines rather than of two
    # workloads. Derived from the shipped reference pair, not a stored scalar.
    ref = effective_rate(prod.r.work, REF_CELL_COST)
    note(@sprintf("at production settings that is %.1fM cell-updates/s", rate / 1e6))
    (cost = cost, rate = rate, ref_rate = ref, worst_residual = worst,
     rate_work = prod.r.work, full_size_scale = scale,
     points = [Dict("name" => pt.name, "sweep_s" => pt.r.sweep_s,
                    "ns_per_cell" => pt.ns, "cfl" => Float64(pt.p.cfl),
                    "tau_max" => Float64(pt.p.tau_max),
                    "lookaheads" => pt.r.work.lookaheads,
                    "substeps" => pt.r.work.substeps,
                    "probes" => pt.r.work.probes,
                    "residual" => resid[i]) for (i, pt) in enumerate(pts)],
     cells = prod.r.cells, sweeps = prod.r.sweeps, seconds = prod.r.seconds,
     n = collect(Int.(g.n)))
end

"""Sequential write and read rate of the scratch directory."""
function measure_disk(dir::String)
    path = joinpath(dir, "PEREGRINE_BENCH.SCRATCH")
    nbytes = Int64(2) * 1024 * 1024 * 1024
    buf = Vector{UInt8}(undef, 64 * 1024 * 1024)
    fill!(buf, 0x5a)
    io = open(path, "w+")
    t0 = time()
    for _ in 1:(nbytes ÷ length(buf))
        write(io, buf)
    end
    flush(io); wrate = nbytes / (time() - t0)
    seek(io, 0)
    t1 = time()
    while !eof(io)
        readbytes!(io, buf, length(buf))
    end
    rrate = nbytes / (time() - t1)
    close(io); rm(path; force = true)
    # The read follows the write immediately, so on any machine with spare
    # RAM it is answered by the page cache and measures memory, not disk.
    # Reported because it is the truth about a warm read, but the smaller of
    # the two is what goes forward as `disk_rate` -- erring toward the slow
    # side is the direction that makes a halo look expensive, which it is.
    cached = rrate > 2.5 * wrate
    note(@sprintf("write %.0f MB/s, read %.0f MB/s%s", wrate / 1e6, rrate / 1e6,
                  cached ? "  (read served by the page cache)" : ""))
    (write = wrate, read = rrate, cached = cached,
     rate = cached ? wrate : min(wrate, rrate))
end

"""What the prefetch is worth here: the same tiled rounds, both ways."""
function measure_prefetch(dir::String, free::Int64)
    # Deliberately small, and deliberately squeezed: a budget of an eighth of
    # the grid forces a real tiling with a real halo, which is the regime a
    # full-scale run is in. Measuring it on a grid that nearly fits would
    # measure nothing.
    g = sized_grid(min(free * 0.05 / cell_bytes(true), 1.2e8))
    cells = ncells(g)
    c = make_case(g)
    dxy, dh = reach_extent(g, c.m, c.p, c.ctl_h, c.nctl; nangle = 16,
                           nsample = 128)
    hx, hy = halo_cells(g, dxy; margin = 2)
    wx = max(hx, 3); wy = max(hy, 3)
    tp = tile_plan(g, wx, wy, hx, hy; warm = true, dxy_cm = dxy, dh_rad = dh)
    note(@sprintf("grid %s, %d tiles of %dx%d, window %dx%d, %.1fx loaded per updated",
                  string(Int.(g.n)), tp.ntiles, tp.wx, tp.wy, tp.nxl, tp.nyl,
                  tp.amplification))

    function once(pf::Bool, tag)
        s = open_store(cells; path = joinpath(dir, "V_$tag.SCRATCH"))
        ps = open_policy_store(cells; path = joinpath(dir, "P_$tag.SCRATCH"))
        t0 = time()
        r, _ = solve_value_ooc!(s, c.occ, g, c.m, c.seeds, c.ctl_h, c.nctl,
                                c.p, tp; rounds = 6, tol = 0.0,
                                tile_sweeps = 3, warm = true, pstore = ps,
                                prefetch = pf)
        el = time() - t0
        close_store!(s); close_store!(ps)
        (seconds = el, rounds = r)
    end

    once(false, "warm")                    # pay for compilation once
    off = once(false, "off")
    on  = once(true,  "on")
    gain = (off.seconds - on.seconds) / off.seconds
    note(@sprintf("sequential %.2f s, prefetched %.2f s -> \033[1;32m%+.1f%%\033[0m",
                  off.seconds, on.seconds, 100 * gain))
    if gain < 0.02
        note("(little to hide here -- this card is slow next to this disk;")
        note(" the prefetch pays where compute is small against the reads)")
    end
    (off_s = off.seconds, on_s = on.seconds, gain = gain,
     amplification = tp.amplification, tiles = tp.ntiles,
     threads = Threads.nthreads())
end

# --------------------------------------------------------------------------

function main()
    say("machine")
    if !CUDA.functional()
        println(stderr, "no functional CUDA device")
        return 1
    end
    dev = CUDA.device()
    free, total = CUDA.memory_info()
    note("gpu     $(CUDA.name(dev))")
    note(@sprintf("vram    %.1f GB total, %.1f GB free", total / 2^30,
                  free / 2^30))
    note("driver  $(CUDA.driver_version())   runtime $(CUDA.runtime_version())")
    note("julia   $(VERSION), $(Threads.nthreads()) threads")
    # Host RAM decides whether the prefetch can hold a full-budget tile --
    # the window buffer is the size of the VRAM budget, so a box whose RAM is
    # not comfortably larger than its card has to shrink the tile to afford
    # it. Without this the answer has to be asked for separately.
    ram_total = Sys.total_memory()
    ram_free = Sys.free_memory()
    note(@sprintf("ram     %.1f GB total, %.1f GB free  (card is %.1f GB, and the prefetch needs a window that size)",
                  ram_total / 2^30, ram_free / 2^30, total / 2^30))
    if Threads.nthreads() < 2
        note("\033[1;33mstart julia with -t auto or the prefetch cannot be measured\033[0m")
    end

    dir = get(ENV, "PEREGRINE_SCRATCH", mktempdir())
    mkpath(dir)

    say("cell cost")
    cr = measure_cell_cost(Int64(free))

    say("disk")
    dk = measure_disk(dir)

    say("prefetch")
    pf = try
        measure_prefetch(dir, Int64(free))
    catch e
        note("skipped: " * first(sprint(showerror, e), 200))
        nothing
    end

    out = Dict{String,Any}(
        "gpu" => CUDA.name(dev),
        "vram_total_bytes" => total,
        "vram_free_bytes" => free,
        "driver_version" => string(CUDA.driver_version()),
        "julia_threads" => Threads.nthreads(),
        "ram_total_bytes" => Sys.total_memory(),
        "ram_free_bytes" => Sys.free_memory(),
        # What `plan` actually prices with. `cell_rate` is the same
        # measurement collapsed to the one number the config used to carry,
        # quoted at production settings; it is a fallback, not the answer.
        "cell_cost" => Dict("per_lookahead_s" => cr.cost.per_lookahead_s,
                            "per_step_s" => cr.cost.per_step_s),
        "cell_cost_fit" => Dict("worst_residual" => cr.worst_residual,
                                "full_size_scale" => cr.full_size_scale,
                                "points" => cr.points),
        "cell_rate" => cr.rate,
        # What that rate is a rate OF. Without it a scalar rate is ambiguous
        # -- this file used to measure at `cfl` 1 / `tau_max` 0.2 and now
        # measures at production settings, and the same number means very
        # different machines under those two readings.
        "cell_rate_work" => Dict("lookaheads" => cr.rate_work.lookaheads,
                                 "substeps" => cr.rate_work.substeps,
                                 "probes" => cr.rate_work.probes),
        "cell_rate_detail" => Dict("cells" => cr.cells, "sweeps" => cr.sweeps,
                                   "seconds" => cr.seconds, "n" => cr.n,
                                   "regime" => "cfl 2, tau_max 0.5"),
        "disk_rate" => dk.rate,
        "disk_write_rate" => dk.write,
        "disk_read_rate_raw" => dk.read,
        "disk_read_was_cached" => dk.cached,
        "prefetch" => pf === nothing ? nothing : Dict(
            "sequential_s" => pf.off_s, "prefetched_s" => pf.on_s,
            "gain" => pf.gain, "amplification" => pf.amplification,
            "tiles" => pf.tiles),
        "reference_cell_rate" => cr.ref_rate,
        "speedup_vs_reference" => cr.rate / cr.ref_rate)

    say("result")
    note(@sprintf("\033[1;32m%.2fx the desktop card this project was tuned on\033[0m",
                  cr.rate / cr.ref_rate))
    note("put these in your solver config and every estimate sharpens:")
    note(@sprintf("    \"cell_cost\": {\"per_lookahead_s\": %.4g, \"per_step_s\": %.4g},",
                  cr.cost.per_lookahead_s, cr.cost.per_step_s))
    note(@sprintf("    \"disk_rate\": %.4g", dk.rate))
    note("(the wizard's `benchmark` subcommand saves them for you)")
    println("\nBENCHMARK ", JSON3.write(out))
    return 0
end

exit(main())
