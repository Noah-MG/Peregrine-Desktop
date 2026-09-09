#!/usr/bin/env julia
#
# Does the honing PID on the card actually land the robot?
#
#   julia --project=solver solver/bench/honing.jl <config.json>
#   julia --project=solver solver/bench/honing.jl --example
#
# Derives the gains the way `MODEL.JSON` does, then drives the full nonlinear
# drivetrain with them from a handoff state -- nominal, and with the fit
# deliberately wrong. Exits non-zero if any case misses its budget, so it can
# gate a card.
#
# Only the `regression`, `zero_c` and `honing_*` keys of the config are read:
# the gains do not depend on the grid, the field or the targets, so neither
# does this. That is what makes it cheap enough to run while tuning.
#
# `--example` uses the model from TABLE_FORMAT.md section 8.7 instead of a
# config, so the check runs without a field, a target list or a solve.

include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "honing_rollout.jl"))

"""The worked model from TABLE_FORMAT.md section 8.7.

Kept here so the honing check has something to run against with no calibration
log to hand, and so that the section 8.8 worked example can be regenerated
rather than hand-copied.
"""
function example_model()
    PS.Model((318.44f0, 6.12f0, -4.87f0,
              -5.33f0, 241.07f0, 3.94f0,
              0.128f0, -0.061f0, 13.706f0),
             (-2.418f0, 0.061f0, 1.472f0,
              0.037f0, -3.106f0, -0.884f0,
              0.0021f0, 0.0009f0, -4.233f0),
             (0.0f0, 0.0f0, 0.0f0),
             (-21.66f0, -0.42f0, 0.53f0,
              -0.31f0, -27.94f0, -0.18f0,
              0.004f0, -0.002f0, -1.882f0),
             (-0.00417f0, 0.00008f0, 0.0312f0,
              0.00011f0, -0.00583f0, -0.0204f0,
              0.0f0, 0.0f0, -0.2461f0),
             (0.0f0, 0.0f0, 0.0f0), (5.0f0, 5.0f0, 0.15f0), 0.55f0)
end

function main()
    if isempty(ARGS)
        println(stderr, "usage: honing.jl <config.json> | --example")
        return 2
    end
    m, cfg = if ARGS[1] == "--example"
        println("model: TABLE_FORMAT.md section 8.7 example\n")
        example_model(), Dict()
    else
        cfg = PS.readjson(ARGS[1], Dict)
        println("model: ", String(cfg["regression"]), "\n")
        load_honing_model(cfg), cfg
    end
    honing_report(m, cfg) ? 0 : 1
end

"""The model the gains come from, and nothing else.

Deliberately not `load_case`: the honing loop never touches the grid, the
field or the occupancy, and building them would make a check that answers in
a moment take as long as a plan does. It is also what lets the wizard offer
this straight after a fit, with no field file in the workspace yet.
"""
function load_honing_model(cfg)
    m, _ = PS.load_model(String(cfg["regression"]))
    Bool(PS.getc(cfg, "zero_c", true)) || return m
    PS.Model(m.B, m.A, m.q, m.S, m.D, (0.0f0, 0.0f0, 0.0f0), m.eps, m.knee)
end

exit(main())
