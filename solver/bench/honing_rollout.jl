# Honing-stage rollout: does the PID the card ships actually land the robot?
#
# Included into harness.jl. The value tables stop at the handoff region, and
# `MODEL.JSON`'s `honing` block carries the gains that take it the rest of the
# way. Those gains are derived from a *linearisation* of the drivetrain -- the
# quadratic drag dropped, the Coulomb term straightened out inside its band,
# the traction knee reduced to a single operating-point gain. This file is
# where that linearisation is held to account, by driving the full nonlinear
# `body_accel` with the gains and measuring where the robot stops.
#
# Nothing here is needed to produce a card. It is the test that the numbers on
# the card are worth shipping.

"""
Run the honing controller against the real nonlinear model.

Deliberately simulated the way the robot runs it, not the way it was derived:

- the controller ticks at `loop_hz` while the plant integrates at `dt`, so
  sampling delay is present rather than assumed away;
- the integrator is clamped to `integral_limit` and frozen while the command
  is saturated. Both are required -- without them the integrator walks the
  command out of the octahedron and the loop never settles;
- the command is clipped to the octahedron `norm1(u) <= 1`, because the wheel
  powers clip whatever the controller asks for;
- `plant` is a separate `Model` from the one the gains came from, so a fit
  error can be injected and the loop scored under it.

`e0` is the body-frame error at handoff, `v0` the body-frame velocity there.
Returns settling time, final error, peak commanded 1-norm, worst overshoot as
a fraction of the initial error, and the share of ticks that clipped.
"""
function honing_rollout(h::Dict, plant::PS.Model;
                        e0 = [15.0, 15.0, 0.25], v0 = [0.0, 0.0, 0.0],
                        tol = [0.5, 0.5, 0.01], T = 8.0, dt = 0.001)
    Kp, Ki, Kd = h["Kp"], h["Ki"], h["Kd"]
    ilim = h["integral_limit"]["value"]
    budget = h["config"]["budget"]
    period = 1.0 / h["config"]["loop_hz"]

    e = collect(float(e0)); v = collect(float(v0)); acc = zeros(3)
    u = zeros(3); tnext = 0.0
    peak = 0.0; over = 0.0; settle = NaN; clipped = 0; ticks = 0

    for k in 1:round(Int, T / dt)
        t = k * dt
        if t >= tnext
            tnext += period
            # Conditional integration: winding up while already saturated
            # only buys command the robot cannot deliver.
            if sum(abs, u) < budget * 0.999
                acc .= clamp.(acc .+ e .* period, -ilim, ilim)
            end
            de = -v                      # d(target - p)/dt with a fixed target
            u = [sum(Kp[r][c] * e[c] + Ki[r][c] * acc[c] + Kd[r][c] * de[c]
                     for c in 1:3) for r in 1:3]
            peak = max(peak, sum(abs, u))
            n1 = sum(abs, u)
            ticks += 1
            if n1 > 1.0                  # the octahedron is a hard limit
                u = u ./ n1
                clipped += 1
            end
        end
        a = PS.body_accel(plant, u[1], u[2], u[3], v[1], v[2], v[3])
        v .+= [a[1], a[2], a[3]] .* dt
        e .-= v .* dt
        over = max(over, maximum(-(e ./ e0)))
        isnan(settle) && all(abs.(e) .< tol) && (settle = t)
        all(isfinite, e) || return (settle = NaN, err = fill(Inf, 3),
                                    peak = Inf, over = Inf, clip = 1.0)
    end
    (settle = settle, err = abs.(e), peak = peak, over = over,
     clip = ticks == 0 ? 0.0 : clipped / ticks)
end

"""Scale a model's drag (`A` and `S` together) and authority (`B`).

The two ways a fit is wrong that the honing loop actually cares about: a
floor grippier or slicker than the calibration run, and a battery that is
not the one the run was made on.
"""
function perturb(m::PS.Model; drag = 1.0, authority = 1.0)
    f = Float32(drag); b = Float32(authority)
    PS.Model(ntuple(i -> m.B[i] * b, 9), ntuple(i -> m.A[i] * f, 9), m.q,
             ntuple(i -> m.S[i] * f, 9), m.D, m.c, m.eps, m.knee)
end

"""Print the honing report for one model: nominal, other handoff states, and
the robustness sweep. `verdict` is false if any case fails its budget."""
function honing_report(m::PS.Model, cfg = Dict(); io = stdout)
    h = PS.honing_gains(m, cfg)
    ok = true
    @printf(io, "omega        = %.4f  (bound by %s: sat %.4f, loop %.4f)\n",
            h["omega"], h["omega_limits"]["bound_by"],
            h["omega_limits"]["saturation"], h["omega_limits"]["loop_rate"])
    @printf(io, "budget       = %.4f   traction gain %.4f   fine band %.2f cm\n",
            h["config"]["budget"], h["traction_gain"]["value"],
            h["fine_band"]["cm"])
    println(io, "third poles  = ", round.(h["poles"]["third"], digits = 3))
    println(io, "kd diagonal  = ", round.(h["poles"]["kd_diag"], digits = 4))
    println(io, "int. limit   = ", round.(h["integral_limit"]["value"], digits = 4))

    # Negative derivative gain means the controller is cancelling the
    # drivetrain's own friction. The design rules it out by construction, so
    # seeing one here means the derivation regressed.
    if any(h["poles"]["kd_diag"] .< 0)
        println(io, "FAIL: negative kd -- the controller is de-damping the robot")
        ok = false
    end

    hdr() = println(io, rpad("  case", 26), rpad("settle", 8), rpad("|err|", 30),
                    rpad("peak", 8), rpad("over", 8), "clip")
    function row(lbl, r; max_err = 0.15, max_over = 0.30, max_settle = 4.0)
        bad = !(all(r.err .< [max_err, max_err, max_err / 50]) &&
                r.over <= max_over &&
                (isfinite(r.settle) && r.settle <= max_settle))
        bad && (ok = false)
        println(io, rpad("  " * lbl, 26),
                rpad(isnan(r.settle) ? "never" : string(round(r.settle, digits = 2)), 8),
                rpad(string(round.(r.err, digits = 4)), 30),
                rpad(string(round(r.peak, digits = 2)), 8),
                rpad(string(round(r.over, digits = 3)), 8),
                string(round(100 * r.clip, digits = 1)) * "%",
                bad ? "   <-- FAIL" : "")
    end

    println(io, "\nhandoff states:"); hdr()
    row("worst-case corner", honing_rollout(h, m))
    row("near zero error",   honing_rollout(h, m; e0 = [1.0, 1.0, 0.02]))
    row("heading only",      honing_rollout(h, m; e0 = [0.1, 0.1, 0.25]))
    row("residual velocity", honing_rollout(h, m; v0 = [40.0, -20.0, 1.0]))

    println(io, "\nfit error -- drag:"); hdr()
    for f in (1.5, 1.3, 1.15, 0.85, 0.7)
        row("drag x$f", honing_rollout(h, perturb(m; drag = f)))
    end
    println(io, "\nfit error -- authority:"); hdr()
    for f in (1.2, 1.1, 0.9, 0.8)
        row("A_u x$f", honing_rollout(h, perturb(m; authority = f)))
    end
    println(io, "\nverdict: ", ok ? "PASS" : "FAIL")
    ok
end
