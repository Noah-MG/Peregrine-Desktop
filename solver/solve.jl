#!/usr/bin/env julia
#
# CLI entry point for the Peregrine value-function solver.
#
#   julia --project=solver solver/solve.jl config.json [plan|solve]
#
# `plan` reports grid size, table bytes and whether it fits in VRAM, then
# exits without solving -- the wizard uses it to let the user tune resolution
# before committing to a long run. `solve` does the work, streaming
# `PROGRESS {json}` lines on stdout.

include(joinpath(@__DIR__, "src", "PeregrineSolver.jl"))
using .PeregrineSolver
using JSON3

function main()
    if isempty(ARGS)
        println(stderr, "usage: solve.jl <config.json> [plan|solve]")
        println(stderr, "       solve.jl --self-test")
        return 2
    end
    ARGS[1] == "--self-test" && return PeregrineSolver.self_test()
    cfgpath = ARGS[1]
    mode = length(ARGS) >= 2 ? ARGS[2] : "solve"
    isfile(cfgpath) || (println(stderr, "no such config: $cfgpath"); return 2)
    cfg = PeregrineSolver.readjson(cfgpath, Dict)

    try
        if mode == "plan"
            println("PLAN ", JSON3.write(PeregrineSolver.plan(cfg)))
        elseif mode == "solve"
            PeregrineSolver.run_solve(cfg)
        else
            println(stderr, "unknown mode '$mode' (expected plan or solve)")
            return 2
        end
    catch e
        println(stderr, "ERROR ", sprint(showerror, e))
        for l in stacktrace(catch_backtrace())[1:min(8, end)]
            println(stderr, "  ", l)
        end
        return 1
    end
    flush(stdout)
    return 0
end

exit(main())
