#!/usr/bin/env julia
#
# Measure what a machine is actually worth to this solver, in the two rates
# `plan` estimates wall clocks from, plus what the prefetch buys on it.
#
#   julia --project=solver -t auto solver/cloud/benchmark.jl [--seconds 30]
#
# Prints a JSON blob on the last line. Everything before it is commentary.
#
# Why this exists: `plan` costs a run from `cell_rate` and `disk_rate`, and
# both defaults are measurements of one 8 GB desktop card and one SATA SSD.
# On any other machine the estimate is an extrapolation, and the whole point
# of the estimate is to decide how long to rent for. Ten minutes here
# replaces the guess with a number.
#
# It needs no field, no targets and no regression: the model is the same
# synthetic fit the self-test uses, so the answer is a property of the
# machine rather than of a particular workspace.

include(joinpath(@__DIR__, "..", "src", "PeregrineSolver.jl"))
using .PeregrineSolver
using .PeregrineSolver: Grid6, Params, Model, ncells, control_set, solve_value!,
                        solve_value_ooc!, target_cells, reach_extent,
                        halo_cells, tile_plan, open_store, open_policy_store,
                        close_store!, _realistic, cell_bytes
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

function make_case(g::Grid6)
    m = _realistic()
    n = g.n
    occ = falses(Int(n[1]) * Int(n[2]) * Int(n[3]))
    # One slab, so the swept collision probe is exercised rather than skipped.
    lo = max(1, Int(n[1]) ÷ 3); hi = min(Int(n[1]) - 1, lo + 3)
    for i in lo:hi, j in 0:(Int(n[2]) - 1), k in 0:(Int(n[3]) - 1)
        occ[(i * Int(n[2]) + j) * Int(n[3]) + k + 1] = true
    end
    p = Params(dt = 0.05f0, nsub = Int32(4), checks = Int32(2),
               adaptive_checks = true, cap = 60.0f0, ntau = Int32(5),
               tau_ratio = 2.0f0, cfl = 1.0f0, tau_min = 0.004f0,
               tau_max = 0.2f0, hmax = 0.0125f0, ncoarse = Int32(8),
               rounds = Int32(2), delta0 = 0.35f0, rk2 = true, simplex = true)
    ctl_t = control_set(4)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]
    targ = (280.0f0, 175.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    ttol = NTuple{6,Float32}(Float32[g.step[k] * 0.5f0 for k in 1:6])
    seeds = Int64.(target_cells(g, targ, ttol)) .+ 1
    (m = m, occ = Vector{Bool}(occ), p = p, nctl = nctl, ctl_h = ctl_h,
     seeds = seeds)
end

"""Cell updates a second, from a real in-core solve on a real grid."""
function measure_cell_rate(free::Int64)
    # A quarter of what is free, at the 16 bytes an in-core cell costs (V plus
    # the Float32 warm-start policy). A quarter rather than all of it because
    # this is a rate measurement, not a capacity test, and a grid that only
    # just fits would measure the allocator.
    want = min(free * 0.25 / cell_bytes(true), 6.0e8)
    g = sized_grid(want)
    cells = ncells(g)
    c = make_case(g)
    note(@sprintf("grid %s = %d cells (%.2f GB in core)",
                  string(Int.(g.n)), cells, cells * cell_bytes(true) / 2^30))

    V = CUDA.fill(c.p.cap, cells)
    pol = CUDA.zeros(Int8, 3 * cells)   # the layout production allocates
    docc = CuArray(c.occ); dctl = CuArray(c.ctl_h)
    dseed = CuArray(c.seeds); total = CUDA.zeros(Float32, 1)

    # One sweep to pay for compilation and the first-touch of every page.
    solve_value!(V, docc, g, c.m, dseed, dctl, c.nctl, c.p; iters = 1,
                 tol = 0.0, use_gpu = true, pol = pol, total = total)
    CUDA.synchronize()

    t1 = time(); solve_value!(V, docc, g, c.m, dseed, dctl, c.nctl, c.p;
                              iters = 1, tol = 0.0, use_gpu = true,
                              pol = pol, total = total)
    CUDA.synchronize(); one_s = time() - t1
    iters = clamp(round(Int, WANT_S / max(one_s, 1e-6)), 3, 200)
    note(@sprintf("one sweep %.3f s -- timing %d more", one_s, iters))

    t0 = time()
    solve_value!(V, docc, g, c.m, dseed, dctl, c.nctl, c.p; iters = iters,
                 tol = 0.0, use_gpu = true, pol = pol, total = total)
    CUDA.synchronize()
    el = time() - t0
    rate = cells * iters / el
    CUDA.unsafe_free!(V); CUDA.unsafe_free!(pol)
    CUDA.unsafe_free!(docc); CUDA.unsafe_free!(dctl); CUDA.unsafe_free!(dseed)
    note(@sprintf("\033[1;32m%.1fM cell-updates/s\033[0m  (%d cells x %d sweeps in %.1f s)",
                  rate / 1e6, cells, iters, el))
    (rate = rate, cells = cells, sweeps = iters, seconds = el,
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

    say("cell rate")
    cr = measure_cell_rate(Int64(free))

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
        "cell_rate" => cr.rate,
        "cell_rate_detail" => Dict("cells" => cr.cells, "sweeps" => cr.sweeps,
                                   "seconds" => cr.seconds, "n" => cr.n),
        "disk_rate" => dk.rate,
        "disk_write_rate" => dk.write,
        "disk_read_rate_raw" => dk.read,
        "disk_read_was_cached" => dk.cached,
        "prefetch" => pf === nothing ? nothing : Dict(
            "sequential_s" => pf.off_s, "prefetched_s" => pf.on_s,
            "gain" => pf.gain, "amplification" => pf.amplification,
            "tiles" => pf.tiles),
        "reference_cell_rate" => 23.4e6,
        "speedup_vs_reference" => cr.rate / 23.4e6)

    say("result")
    note(@sprintf("\033[1;32m%.2fx the desktop card this project was tuned on\033[0m",
                  cr.rate / 23.4e6))
    note("put these two in your solver config and every estimate sharpens:")
    note(@sprintf("    \"cell_rate\": %.4g,", cr.rate))
    note(@sprintf("    \"disk_rate\": %.4g", dk.rate))
    println("\nBENCHMARK ", JSON3.write(out))
    return 0
end

exit(main())
