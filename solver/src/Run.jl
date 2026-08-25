# Orchestration: plan, solve every target, write tables and manifest.
# Included into the PeregrineSolver module.

const DEFAULT_VMAX = 150.0     # cm/s   -- a brisk but ordinary FTC chassis
const DEFAULT_WMAX = 10.0      # rad/s  -- ~1.6 rev/s, matches the spin logs

"""Emit one machine-readable progress line for the Python wizard."""
function progress(; kw...)
    println("PROGRESS ", JSON3.write(Dict(pairs(kw))))
    flush(stdout)
end

getc(cfg, key, default) = haskey(cfg, key) && cfg[key] !== nothing ? cfg[key] : default

"""
Build the grid from the field bounds plus the velocity envelope.

Heading always spans a full turn, so its bounds are implicit.
"""
function build_grid(cfg, bounds)
    n = NTuple{6,Int}(Int.(getc(cfg, "n", [41, 41, 16, 11, 11, 11])))
    vmax = Float64(getc(cfg, "vmax", DEFAULT_VMAX))
    wmax = Float64(getc(cfg, "wmax", DEFAULT_WMAX))
    lo = (bounds[1], bounds[2], -π, -vmax, -vmax, -wmax)
    hi = (bounds[3], bounds[4],  π,  vmax,  vmax,  wmax)
    Grid6(n, lo, hi)
end

"""Size and feasibility report, cheap enough to run before committing."""
function plan(cfg::AbstractDict)
    bounds, _, _, _ = load_field(String(cfg["field"]))
    names, _, _ = load_targets(String(cfg["targets"]))
    g = build_grid(get(cfg, "grid", Dict()), bounds)
    dtype = String(getc(cfg, "dtype", "u16"))
    haskey(DTYPES, dtype) || error("unknown dtype '$dtype'")
    eb = DTYPES[dtype].bytes
    cells = ncells(g)
    per = cells * eb
    chunk_elements = Int(getc(cfg, "chunk_elements", 1 << 23))
    ok, need, free = gpu_fits(g)
    Dict(
        "cells" => cells,
        "axes" => collect(AXES),
        "n" => collect(Int.(g.n)),
        "dtype" => dtype,
        "elem_bytes" => eb,
        "bytes_per_target" => per,
        "n_targets" => length(names),
        "bytes_total" => per * length(names),
        "chunks_per_target" => cld(cells, chunk_elements),
        "gpu_available" => CUDA.functional(),
        "gpu_fits" => ok,
        "gpu_need_gb" => need,
        "gpu_free_gb" => free,
    )
end

"""
Build the dynamics model the robot needs alongside the tables.

Written in a deliberately over-general form:

    a = A_s*s + A_u*u + A_ss*(s.*s) + A_uu*(u.*u)
        + A_sgn*csign(s) + A_absv*(abs(s).*s) + k

Several blocks are usually zero -- the fitted regression has no position
dependence and no control-squared terms. They are emitted anyway so that the
regression can gain or lose terms without the robot-side reader changing: it
always multiplies the same things and adds them up. `nonzero_blocks` says
which are actually carrying anything.

The matrices describe the model **as the solver actually used it**, so if
`zero_c` was set the constant here is zero too. Otherwise the robot's
dynamics would disagree with the tables that were solved from them.
"""
function model_json(m::Model, reg_path::AbstractString, zeroed_c::Bool)
    row3(t, r) = [Float64(t[(r - 1) * 3 + c]) for c in 1:3]
    Am = [row3(m.A, r) for r in 1:3]      # velocity -> acceleration
    Bm = [row3(m.B, r) for r in 1:3]      # control  -> acceleration

    # State is the same 6-vector the value tables use. Position and heading do
    # not affect the dynamics, so those columns are zero.
    A_s  = [[0.0, 0.0, 0.0, Am[r][1], Am[r][2], Am[r][3]] for r in 1:3]
    # Only omega^2 is fitted, and it lands in the w column of s.*s.
    A_ss = [[0.0, 0.0, 0.0, 0.0, 0.0, Float64(m.q[r])] for r in 1:3]
    A_u  = Bm
    A_uu = [[0.0, 0.0, 0.0] for _ in 1:3]
    Sm   = [row3(m.S, r) for r in 1:3]
    Dm   = [row3(m.D, r) for r in 1:3]
    A_sgn = [[0.0, 0.0, 0.0, Sm[r][1], Sm[r][2], Sm[r][3]] for r in 1:3]
    A_abs = [[0.0, 0.0, 0.0, Dm[r][1], Dm[r][2], Dm[r][3]] for r in 1:3]
    k    = [Float64(m.c[r]) for r in 1:3]

    Dict(
        "schema_version" => 2,
        "generator" => "peregrine-desktop",
        "regression_sha256" => filehash(reg_path),
        "equation" => "a = A_s*s + A_u*u + A_ss*(s.*s) + A_uu*(u.*u) + " *
                      "A_sgn*csign(s) + A_absv*(abs(s).*s) + k",
        "csign" => Dict(
            "formula" => "csign(s)_i = clamp(s_i / coulomb_eps_i, -1, 1)",
            "coulomb_eps" => [0.0, 0.0, 0.0, Float64(m.eps[1]),
                              Float64(m.eps[2]), Float64(m.eps[3])],
            "why" => "Coulomb friction is smoothed instead of using a hard " *
                     "sign(), which would flip discontinuously at zero and " *
                     "make an integrator chatter. Use exactly this form.",
        ),
        "output" => Dict(
            "vector" => ["a_x", "a_y", "alpha"],
            "units" => ["cm/s^2", "cm/s^2", "rad/s^2"],
            "frame" => "robot body",
        ),
        "state" => Dict(
            "vector" => collect(AXES),
            "units" => ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
            "frame_note" => "IMPORTANT: vx and vy must be rotated into the " *
                "ROBOT BODY frame before use here. The value tables index " *
                "field-frame velocity, so rotate by -h between the two. " *
                "Motor forces act along the body axes, which is why the " *
                "model cannot be written with constant matrices in the " *
                "field frame. x, y and h have zero coefficients throughout.",
        ),
        "control" => Dict(
            "vector" => ["fwd", "strafe", "turn"],
            "note" => "mecanum projection of the wheel powers; the " *
                      "admissible set is |fwd| + |strafe| + |turn| <= 1",
            "saturation" => Dict(
                "knee" => m.knee > 0 ? Float64(m.knee) : nothing,
                "formula" => "m = norm(u_raw); " *
                             "u = u_raw * tanh(m/knee) / (m/knee)",
                "why" => "Past the knee the tyres stop delivering, so extra " *
                         "command buys no extra force. APPLY THIS BEFORE " *
                         "A_u -- the gains were fitted against the saturated " *
                         "command. A null knee means no saturation.",
            ),
        ),
        "A_s" => A_s,      # 3x6
        "A_u" => A_u,      # 3x3
        "A_ss" => A_ss,    # 3x6
        "A_uu" => A_uu,    # 3x3
        "A_sgn" => A_sgn,  # 3x6, Coulomb, multiplies csign(s)
        "A_absv" => A_abs, # 3x6, quadratic drag, multiplies abs(s).*s
        "k" => k,          # 3
        "constant_zeroed" => zeroed_c,
        "nonzero_blocks" => [b for (b, nz) in (
            ("A_s", any(any(!iszero, r) for r in A_s)),
            ("A_u", any(any(!iszero, r) for r in A_u)),
            ("A_ss", any(any(!iszero, r) for r in A_ss)),
            ("A_uu", any(any(!iszero, r) for r in A_uu)),
            ("A_sgn", any(any(!iszero, r) for r in A_sgn)),
            ("A_absv", any(any(!iszero, r) for r in A_abs)),
            ("k", any(!iszero, k))) if nz],
    )
end

"""
Solve every target and write the card image.

Tables are written into `out_dir/TABLES`, and the manifest to
`out_dir/MANIFEST.JSON`, mirroring the layout the SD card will have so the
wizard can copy the tree across verbatim.
"""
function run_solve(cfg::AbstractDict)
    reg_path = String(cfg["regression"])
    field_path = String(cfg["field"])
    targ_path = String(cfg["targets"])
    out_dir = String(cfg["out_dir"])

    m, regcfg = load_model(reg_path)
    if Bool(getc(cfg, "zero_c", true))
        # A drivetrain at rest with no power must not accelerate. A nonzero
        # constant makes the simulated robot drift forever, so it is zeroed
        # by default -- it is a diagnostic of the fit, not real dynamics.
        m = Model(m.B, m.A, m.q, m.S, m.D, (0.0f0, 0.0f0, 0.0f0),
                  m.eps, m.knee)
    end
    bounds, polys, robot, fieldcfg = load_field(field_path)
    names, states, _ = load_targets(targ_path)

    gcfg = get(cfg, "grid", Dict())
    g = build_grid(gcfg, bounds)
    cells = ncells(g)

    dtype = String(getc(cfg, "dtype", "u16"))
    haskey(DTYPES, dtype) || error("unknown dtype '$dtype'")
    scale = Float64(getc(cfg, "scale", DTYPES[dtype].scale))
    chunk_elements = Int(getc(cfg, "chunk_elements", 1 << 23))
    ispow2(chunk_elements) || error("chunk_elements must be a power of two")

    dt = Float32(getc(cfg, "dt", 0.05))
    nsub = Int(getc(cfg, "substeps", 4))
    checks = Int(getc(cfg, "sweep_checks", 3))
    nearest = Bool(getc(cfg, "nearest", false))
    iters = Int(getc(cfg, "iterations", 400))
    tol = Float64(getc(cfg, "tolerance", 1e-3))
    # 3 samples the octahedron boundary properly. Corners alone (level 1)
    # made every cell pessimistic by ~1.8 s on a real model, because a plain
    # diagonal was not in the control set at all.
    level = Int(getc(cfg, "control_level", 3))
    margin = Float64(getc(cfg, "margin_cm", 0.0))
    # Safety gap held around every obstacle, applied before the robot's own
    # footprint is swept in.
    clearance = Float64(getc(cfg, "clearance_cm", 5.0))
    # Half a cell, so the seed is normally the single nearest cell. The online
    # optimizer owns the real arrival test; this only has to plant the seed.
    ttol = NTuple{6,Float32}(Float32.(getc(cfg, "target_tol",
              [g.step[k] * 0.5 for k in 1:6])))
    # Unreached cells hold this finite value rather than Inf; see `interp`.
    # It also bounds what the table can express, so keep it inside the dtype's
    # range: u16 at 1 ms tops out at 65.5 s.
    cap = Float32(getc(cfg, "value_cap", 60.0))

    ctl_t = control_set(level)
    nctl = length(ctl_t)
    ctl_h = Float32[getindex.(ctl_t, 1); getindex.(ctl_t, 2); getindex.(ctl_t, 3)]

    progress(phase = "setup", cells = cells, n = collect(Int.(g.n)),
             controls = nctl, dtype = dtype,
             bytes_per_target = cells * DTYPES[dtype].bytes,
             n_targets = length(names))

    hsub = Int(getc(cfg, "heading_substeps", 3))
    occ_h = build_occupancy(g, polys, margin, robot, hsub, clearance)
    blocked = count(occ_h)
    progress(phase = "occupancy", blocked_cells = blocked,
             blocked_frac = blocked / length(occ_h),
             robot_vertices = robot === nothing ? 0 : size(robot, 2),
             heading_substeps = hsub, clearance_cm = clearance)

    backend = String(getc(cfg, "backend", "auto"))
    use_gpu = backend == "cuda" || (backend == "auto" && CUDA.functional())
    if use_gpu
        ok, need, free = gpu_fits(g)
        ok || error("grid needs $(round(need, digits=2)) GB of VRAM but only " *
                    "$(round(free, digits=2)) GB is free; reduce the grid, or " *
                    "set backend to \"cpu\"")
    end
    progress(phase = "backend", backend = use_gpu ? "cuda" : "cpu")

    if use_gpu
        V = CUDA.fill(cap, cells)
        occ = CuArray(occ_h)
        ctl = CuArray(ctl_h)
        total = CUDA.zeros(Float32, 1)
    else
        V = fill(cap, cells)
        occ = occ_h
        ctl = ctl_h
    end

    tables_dir = joinpath(out_dir, "TABLES")
    mkpath(tables_dir)
    entries = Any[]

    for (ti, s) in enumerate(states)
        tcells = target_cells(g, s, ttol)
        tidx = use_gpu ? CuArray(Int64.(tcells) .+ 1) : (Int64.(tcells) .+ 1)

        fill!(V, cap)
        @views V[tidx] .= 0.0f0

        t0 = time()
        last_delta = Inf
        done_iters = 0
        for it in 1:iters
            d = use_gpu ?
                sweep_gpu!(V, occ, g, m, ctl, nctl, dt, nsub, checks, nearest,
                           cap, total) :
                sweep_cpu!(V, occ, g, m, ctl, nctl, dt, nsub, checks, nearest, cap)
            # The seed must be reasserted: a sweep can lower a target cell
            # below zero-cost only through interpolation noise, and letting it
            # drift would corrupt the whole basin.
            @views V[tidx] .= 0.0f0
            done_iters = it
            last_delta = d
            el = time() - t0
            if it % 5 == 0 || it == 1 || d <= tol
                progress(phase = "solve", target = ti - 1, target_name = names[ti],
                         iter = it, iters = iters, delta = d,
                         elapsed_s = el, sweep_s = el / it,
                         eta_s = (iters - it) * el / it)
            end
            d <= tol && break
        end

        Vh = use_gpu ? Array(V) : copy(V)
        reached = count(v -> v < cap, Vh)
        progress(phase = "encode", target = ti - 1, target_name = names[ti],
                 iters_done = done_iters, delta = last_delta,
                 reached_frac = reached / length(Vh))

        info = write_table(tables_dir, ti - 1, Vh, dtype, scale, chunk_elements, cap)
        push!(entries, Dict(
            "index" => ti - 1,
            "name" => names[ti],
            "state" => collect(Float64.(s)),
            "file_pattern" => @sprintf("TABLES/T%02dC%%04d.BIN", ti - 1),
            "n_chunks" => info.nchunks,
            "bytes" => info.bytes,
            "sha256" => info.sha256,
            "iterations" => done_iters,
            "final_delta" => last_delta,
            "reached_frac" => reached / length(Vh),
        ))
        progress(phase = "target_done", target = ti - 1, target_name = names[ti],
                 chunks = info.nchunks, bytes = info.bytes,
                 elapsed_s = time() - t0)
    end

    manifest = Dict(
        "schema_version" => 1,
        "generated_utc" => string(round(now_utc(), Second)) * "Z",
        "generator" => "peregrine-desktop",
        "regression_sha256" => filehash(reg_path),
        "field_sha256" => filehash(field_path),
        "grid" => Dict(
            "axes" => collect(AXES),
            "n" => collect(Int.(g.n)),
            "min" => [Float64(g.lo[k]) for k in 1:6],
            "max" => [Float64(g.lo[k] + g.step[k] * (k == 3 ? g.n[k] : g.n[k] - 1))
                      for k in 1:6],
            "wrap" => [false, false, true, false, false, false],
            "units" => ["cm", "cm", "rad", "cm/s", "cm/s", "rad/s"],
            "frame" => "field",
            "frame_note" => "x, y, h and the vx, vy velocity axes are all " *
                            "field frame; omega is frame independent",
            "total_cells" => cells,
            "index_formula" => "((((ix*Ny+iy)*Nh+ih)*Nvx+ivx)*Nvy+ivy)*Nw+iw",
        ),
        "encoding" => Dict(
            "dtype" => dtype,
            "elem_bytes" => DTYPES[dtype].bytes,
            "scale" => scale,
            "unit" => "seconds",
            "unreachable" => dtype == "u8" ? 255 :
                             dtype == "u16" ? 65535 : "NaN",
            "byte_order" => "little",
            "order" => "row_major_c",
            "chunk_elements" => chunk_elements,
            "chunk_shift" => trailing_zeros(chunk_elements),
        ),
        "solver" => Dict(
            "dt" => Float64(dt), "substeps" => nsub, "controls" => nctl,
            "control_level" => level, "iterations_max" => iters,
            "tolerance" => tol, "nearest" => nearest,
            "sweep_checks" => checks, "zero_c" => Bool(getc(cfg, "zero_c", true)),
            "margin_cm" => margin, "clearance_cm" => clearance,
            "value_cap_s" => Float64(cap),
            "heading_substeps" => hsub,
            "robot_vertices" => robot === nothing ? 0 : size(robot, 2),
            "occupancy" => "per heading bin (x, y, h)",
        ),
        "model_file" => "MODEL.JSON",
        "targets" => entries,
    )
    open(joinpath(out_dir, "MANIFEST.JSON"), "w") do io
        JSON3.pretty(io, manifest)
    end

    mj = model_json(m, reg_path, Bool(getc(cfg, "zero_c", true)))
    open(joinpath(out_dir, "MODEL.JSON"), "w") do io
        JSON3.pretty(io, mj)
    end
    progress(phase = "model", file = "MODEL.JSON",
             nonzero_blocks = mj["nonzero_blocks"])
    progress(phase = "done", out_dir = out_dir, targets = length(entries))
    manifest
end

using Dates
now_utc() = Dates.now(Dates.UTC)

function filehash(path)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end
