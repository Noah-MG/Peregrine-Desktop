# Geometry, IO, solve drivers and table output.
# Included into the PeregrineSolver module.

using TOML

# --------------------------------------------------------------------------
# Occupancy
# --------------------------------------------------------------------------

"""
Read a JSON file, tolerating a UTF-8 BOM.

Windows editors (and PowerShell's own `Set-Content -Encoding utf8`) prepend a
BOM, which is legal UTF-8 but not legal JSON. These files get hand-edited, so
stripping it here is worth more than being strict.
"""
function readjson(path::AbstractString)
    s = read(path, String)
    startswith(s, '﻿') && (s = s[nextind(s, 1):end])
    JSON3.read(s)
end

function readjson(path::AbstractString, ::Type{T}) where {T}
    s = read(path, String)
    startswith(s, '﻿') && (s = s[nextind(s, 1):end])
    JSON3.read(s, T)
end

"""Ray-casting point-in-polygon. `poly` is a 2xN matrix of vertices."""
function inpoly(px::Float64, py::Float64, poly::Matrix{Float64})
    n = size(poly, 2)
    inside = false
    j = n
    @inbounds for i in 1:n
        xi, yi = poly[1, i], poly[2, i]
        xj, yj = poly[1, j], poly[2, j]
        if ((yi > py) != (yj > py)) &&
           (px < (xj - xi) * (py - yi) / (yj - yi + eps()) + xi)
            inside = !inside
        end
        j = i
    end
    inside
end

"""Do segments p1-p2 and p3-p4 properly cross?"""
@inline function seg_cross(p1, p2, p3, p4)
    d(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
    d1 = d(p3, p4, p1); d2 = d(p3, p4, p2)
    d3 = d(p1, p2, p3); d4 = d(p1, p2, p4)
    ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0))
end

"""
Do two polygons overlap? Exact for concave polygons.

Three cases cover it: a vertex of one inside the other (containment either
way), or a pair of edges crossing (partial overlap). Testing directly like
this avoids computing Minkowski sums, which would need the robot and every
obstacle decomposed into convex pieces first.
"""
function polys_overlap(A::Matrix{Float64}, B::Matrix{Float64})
    na, nb = size(A, 2), size(B, 2)
    @inbounds for i in 1:na
        inpoly(A[1, i], A[2, i], B) && return true
    end
    @inbounds for j in 1:nb
        inpoly(B[1, j], B[2, j], A) && return true
    end
    @inbounds for i in 1:na
        a1 = (A[1, i], A[2, i])
        a2 = (A[1, mod1(i + 1, na)], A[2, mod1(i + 1, na)])
        for j in 1:nb
            b1 = (B[1, j], B[2, j])
            b2 = (B[1, mod1(j + 1, nb)], B[2, mod1(j + 1, nb)])
            seg_cross(a1, a2, b1, b2) && return true
        end
    end
    false
end

"""Distance from point `p` to segment `a`-`b`."""
@inline function pt_seg_dist(p, a, b)
    abx = b[1] - a[1]; aby = b[2] - a[2]
    L2 = abx * abx + aby * aby
    t = L2 <= 0 ? 0.0 :
        clamp(((p[1] - a[1]) * abx + (p[2] - a[2]) * aby) / L2, 0.0, 1.0)
    dx = p[1] - (a[1] + t * abx); dy = p[2] - (a[2] + t * aby)
    sqrt(dx * dx + dy * dy)
end

"""Distance between segments `a1`-`a2` and `b1`-`b2`, zero if they cross."""
@inline function seg_seg_dist(a1, a2, b1, b2)
    seg_cross(a1, a2, b1, b2) && return 0.0
    min(pt_seg_dist(a1, b1, b2), pt_seg_dist(a2, b1, b2),
        pt_seg_dist(b1, a1, a2), pt_seg_dist(b2, a1, a2))
end

"""
Is polygon `A` within `clearance` of polygon `B`?

Requiring a gap of `d` around every obstacle is the same as asking whether
the two shapes come within `d` of each other, so the clearance is applied by
measuring distance rather than by offsetting the obstacle polygons. Offsetting
would mean handling miter joins and self-intersections; this is exact, works
on concave polygons, and keeps the clearance a true Euclidean distance rather
than something quantised to the grid.
"""
function polys_too_close(A::Matrix{Float64}, B::Matrix{Float64}, d::Real)
    polys_overlap(A, B) && return true
    d <= 0 && return false
    na, nb = size(A, 2), size(B, 2)
    @inbounds for i in 1:na
        a1 = (A[1, i], A[2, i])
        a2 = (A[1, mod1(i + 1, na)], A[2, mod1(i + 1, na)])
        for j in 1:nb
            b1 = (B[1, j], B[2, j])
            b2 = (B[1, mod1(j + 1, nb)], B[2, mod1(j + 1, nb)])
            seg_seg_dist(a1, a2, b1, b2) < d && return true
        end
    end
    false
end

"""Same test for a point robot."""
function point_too_close(px::Float64, py::Float64, B::Matrix{Float64}, d::Real)
    inpoly(px, py, B) && return true
    d <= 0 && return false
    nb = size(B, 2)
    @inbounds for j in 1:nb
        b1 = (B[1, j], B[2, j])
        b2 = (B[1, mod1(j + 1, nb)], B[2, mod1(j + 1, nb)])
        pt_seg_dist((px, py), b1, b2) < d && return true
    end
    false
end

"""
The heading angles `build_occupancy` rasterises, bin by bin.

Kept as its own function because three things have to agree on it exactly:
the occupancy build, the feasible-bounds calculation that sizes the grid, and
the self-test that checks one against the other. A bin whose bounds came from
one angle set and whose occupancy came from another would show up as a thin
ring of cells that the grid calls legal and the mask calls blocked -- or, far
worse, the other way round.

`nh` bins span the full turn from -pi with step `2pi/nh`, matching
`axisvalue(g, 3, k)`. Each bin is covered by `substeps` angles spanning it,
so a heading between two bin centres cannot sneak through.
"""
function heading_bin_angles(nh::Integer, substeps::Integer)
    step = 2pi / nh
    half = step / 2
    map(0:(Int(nh) - 1)) do k
        c = -pi + k * step
        substeps <= 1 ? [c] :
        [c - half + 2half * (s - 1) / (substeps - 1) for s in 1:Int(substeps)]
    end
end

"""
How far the footprint reaches past the tracking point at heading `theta`, on
each of the four sides: `(xlo, xhi, ylo, yhi)`.

`xlo` is the reach in the *negative* x direction, reported positive, because
that is the form the wall constraint wants: the tracking point has to sit at
least `xlo` inboard of the left wall.
"""
function footprint_reach(robot::Union{Nothing,Matrix{Float64}}, theta::Real)
    (robot === nothing || size(robot, 2) < 3) && return (0.0, 0.0, 0.0, 0.0)
    c, s = cos(theta), sin(theta)
    xlo = -Inf; xhi = -Inf; ylo = -Inf; yhi = -Inf
    @inbounds for i in 1:size(robot, 2)
        rx = c * robot[1, i] - s * robot[2, i]
        ry = s * robot[1, i] + c * robot[2, i]
        xlo = max(xlo, -rx); xhi = max(xhi, rx)
        ylo = max(ylo, -ry); yhi = max(yhi, ry)
    end
    (xlo, xhi, ylo, yhi)
end

"""
The box of tracking-point positions that keep the whole footprint inside the
field, with `clearance` held off the wall, at one heading.

The exterior wall is an obstacle like any other, and this is what "like any
other" has to mean for it. An obstacle gets the robot's own shape swept in
and the safety gap held around it; the field boundary now gets exactly the
same two things, from the inside. Before this existed the boundary was a
constraint on the tracking *point* alone, so a 36 cm chassis could park with
18 cm of itself outside the field -- a rule strictly looser than the one
applied to a wall-hugging obstacle one centimetre inboard of it.

Returned as `(xlo, xhi, ylo, yhi)`. A heading the field cannot accommodate at
all comes back with `xlo` above `xhi`, which the callers test for rather than
silently building an empty grid from.
"""
function wall_box(bounds::NTuple{4,Float64},
                  robot::Union{Nothing,Matrix{Float64}},
                  clearance::Real, theta::Real)
    rxlo, rxhi, rylo, ryhi = footprint_reach(robot, theta)
    d = Float64(clearance)
    (bounds[1] + d + rxlo, bounds[3] - d - rxhi,
     bounds[2] + d + rylo, bounds[4] - d - ryhi)
end

"""
The x/y span the value table actually has to cover.

Every state outside it is one the robot cannot legally hold at any heading,
so storing it would be storing `unreachable` -- and in six dimensions that
saving is not marginal. On a 366 cm field with a 36 cm chassis and 5 cm of
clearance the span drops to 320 cm a side, which is 76% of the cells.

The box is the bounding box of the *union* over heading bins, not their
intersection: a cell only one heading can occupy is still a real state and
has to be in the table. Cells inside the box that no heading can occupy, and
cells legal at some headings but not others, are handled where every other
obstacle is -- per heading bin, in `build_occupancy`.

Computed from the same bin angle sets the occupancy build uses, so the two
cannot drift apart. Each bin is an intersection over its substep angles
(blocked at any of them means blocked for the bin); the bins are then
unioned.
"""
function feasible_bounds(bounds::NTuple{4,Float64},
                         robot::Union{Nothing,Matrix{Float64}},
                         clearance::Real, nh::Integer, substeps::Integer)
    xlo = Inf; xhi = -Inf; ylo = Inf; yhi = -Inf
    nfeas = 0
    for angles in heading_bin_angles(nh, substeps)
        bxlo = -Inf; bxhi = Inf; bylo = -Inf; byhi = Inf
        for th in angles
            a, b, c, d = wall_box(bounds, robot, clearance, th)
            bxlo = max(bxlo, a); bxhi = min(bxhi, b)
            bylo = max(bylo, c); byhi = min(byhi, d)
        end
        (bxlo <= bxhi && bylo <= byhi) || continue      # impossible heading
        nfeas += 1
        xlo = min(xlo, bxlo); xhi = max(xhi, bxhi)
        ylo = min(ylo, bylo); yhi = max(yhi, byhi)
    end
    nfeas == 0 && error(
        "the robot does not fit inside the field at any heading: the field " *
        "is $(round(bounds[3] - bounds[1], digits = 1)) x " *
        "$(round(bounds[4] - bounds[2], digits = 1)) cm, and the footprint " *
        "plus $(clearance) cm of clearance needs more than that. Check the " *
        "units on the robot polygon, or lower the clearance")
    (xlo = xlo, xhi = xhi, ylo = ylo, yhi = yhi, headings = nfeas)
end

"""
Rasterise the configuration-space obstacle onto the (x, y, h) grid.

The C-obstacle of a polygonal robot among polygonal obstacles is **not** a
polyhedron in (x, y, theta). A contact constraint has the form

    n_x*x + n_y*y + A*cos(theta) + B*sin(theta) = c

which is planar only when A = B = 0, i.e. only for a robot whose vertex sits
on the centre of rotation -- a point robot. For a real chassis the faces are
ruled surfaces, and the slice even changes vertex count as it rotates.

What *is* exactly true is that every fixed-theta slice is a polygon. So the
occupancy grid is built per heading bin, which is exact at each sampled
heading and is all the solver ever asks about. Each bin is unioned over
`substeps` angles spanning the bin, so a heading between two samples cannot
sneak through.

`clearance_cm` is a safety gap held around every obstacle, applied *before*
the robot's own shape is considered, so it is a plain "keep this far away"
distance independent of how big the robot is. `margin_cm` is different and
generally left at zero: it dilates the finished grid, which is only useful to
force a very thin obstacle to occupy at least one cell.

**The exterior wall is one of the obstacles.** Pass `bounds` and the boundary
is enforced the same way the polygons are: the rotated footprint has to fit
inside the field with `wall_clearance_cm` to spare, at every angle the bin
covers. The constraint is separable in x and y, so it costs a comparison
rather than a polygon test -- but it is the identical rule, and taking it
from `wall_box` is what keeps it identical. Omit `bounds` and only the
polygons are rasterised, which is how the self-test isolates the two.

Result is Nx*Ny*Nh booleans -- still tiny next to the 6D value table.
"""
function build_occupancy(g::Grid6, polys::Vector{Matrix{Float64}},
                         margin_cm::Real, robot::Union{Nothing,Matrix{Float64}},
                         substeps::Int = 3, clearance_cm::Real = 0.0;
                         bounds::Union{Nothing,NTuple{4,Float64}} = nothing,
                         wall_clearance_cm::Real = clearance_cm)
    nx, ny, nh = Int(g.n[1]), Int(g.n[2]), Int(g.n[3])
    occ = falses(nx, ny, nh)
    point_robot = robot === nothing || size(robot, 2) < 3

    # Bounding radius, to skip cells that cannot possibly touch an obstacle.
    rad = point_robot ? 0.0 : maximum(sqrt.(robot[1, :] .^ 2 .+ robot[2, :] .^ 2))

    # A tolerance, not a fudge, and it has to clear Float32. `feasible_bounds`
    # puts the grid edge exactly on the most permissive heading's limit, so
    # the outermost cell sits *on* the constraint -- and `axisvalue`
    # reconstructs that limit in Float32, whose spacing is 2e-6 cm at 20 and
    # 3e-5 cm at field scale. A tolerance below that blanks the entire outer
    # ring of the table, which is a whole row of legal wall-hugging states
    # thrown away over floating-point noise. Ten microns clears it by two
    # orders of magnitude and is far below anything mechanical.
    tol = 1.0e-3

    for k in 0:nh-1
        θc = Float64(axisvalue(g, 3, k))
        half = Float64(g.step[3]) / 2
        angles = substeps <= 1 ? (θc,) :
                 ntuple(s -> θc - half + 2half * (s - 1) / (substeps - 1),
                        substeps)

        # The wall first. It is separable, so one pass along each axis marks
        # every cell the footprint would hang out of the field at this
        # heading. Blocked at any substep angle means blocked for the bin --
        # the same union over substeps the polygon loop below builds.
        if bounds !== nothing
            for θ in angles
                bxlo, bxhi, bylo, byhi = wall_box(bounds, robot,
                                                  wall_clearance_cm, θ)
                for i in 0:nx-1
                    px = Float64(axisvalue(g, 1, i))
                    (px < bxlo - tol || px > bxhi + tol) || continue
                    for j in 0:ny-1
                        occ[i+1, j+1, k+1] = true
                    end
                end
                for j in 0:ny-1
                    py = Float64(axisvalue(g, 2, j))
                    (py < bylo - tol || py > byhi + tol) || continue
                    for i in 0:nx-1
                        occ[i+1, j+1, k+1] = true
                    end
                end
            end
        end

        for θ in angles
            Rr = point_robot ? zeros(2, 0) :
                 [cos(θ) -sin(θ); sin(θ) cos(θ)] * robot

            for P in polys
                # Only sweep cells near this obstacle.
                pad = rad + margin_cm + clearance_cm
                x0 = minimum(P[1, :]) - pad
                x1 = maximum(P[1, :]) + pad
                y0 = minimum(P[2, :]) - pad
                y1 = maximum(P[2, :]) + pad
                i0 = max(0, floor(Int, (x0 - g.lo[1]) / g.step[1]))
                i1 = min(nx - 1, ceil(Int, (x1 - g.lo[1]) / g.step[1]))
                j0 = max(0, floor(Int, (y0 - g.lo[2]) / g.step[2]))
                j1 = min(ny - 1, ceil(Int, (y1 - g.lo[2]) / g.step[2]))

                for i in i0:i1, j in j0:j1
                    occ[i+1, j+1, k+1] && continue
                    px = Float64(axisvalue(g, 1, i))
                    py = Float64(axisvalue(g, 2, j))
                    if point_robot
                        occ[i+1, j+1, k+1] = point_too_close(px, py, P,
                                                             clearance_cm)
                    else
                        body = Rr .+ [px; py]
                        occ[i+1, j+1, k+1] = polys_too_close(body, P,
                                                             clearance_cm)
                    end
                end
            end
        end
    end

    # Dilate within each heading slice: covers the error from testing only
    # cell centres, and keeps thin obstacles at least one cell thick so the
    # solver's interpolation cannot leak a value across them.
    if margin_cm > 0
        rx = max(0, ceil(Int, margin_cm / g.step[1]))
        ry = max(0, ceil(Int, margin_cm / g.step[2]))
        if rx > 0 || ry > 0
            src = copy(occ)
            for k in 1:nh, i in 1:nx, j in 1:ny
                src[i, j, k] || continue
                for di in -rx:rx, dj in -ry:ry
                    ii, jj = i + di, j + dj
                    (1 <= ii <= nx && 1 <= jj <= ny) || continue
                    occ[ii, jj, k] = true
                end
            end
        end
    end

    # Flatten to match the kernel's  (i1*n2 + i2)*n3 + i3  indexing.
    flat = Vector{Bool}(undef, nx * ny * nh)
    for i in 0:nx-1, j in 0:ny-1, k in 0:nh-1
        flat[(i*ny + j) * nh + k + 1] = occ[i+1, j+1, k+1]
    end
    flat
end

# --------------------------------------------------------------------------
# Input files
# --------------------------------------------------------------------------

"""Load the drivetrain regression written by `fit_drivetrain.py`."""
function load_model(path::AbstractString)
    cfg = TOML.parsefile(path)
    haskey(cfg, "mecanum_basis") ||
        error("$path has no [mecanum_basis]; is it a drivetrain_fit.toml?")
    mb = cfg["mecanum_basis"]
    B = NTuple{9,Float32}(Float32(mb["B"][r][c]) for r in 1:3 for c in 1:3)
    A = NTuple{9,Float32}(Float32(mb["A"][r][c]) for r in 1:3 for c in 1:3)
    q = NTuple{3,Float32}(Float32(v) for v in mb["q"])
    c = NTuple{3,Float32}(Float32(v) for v in mb["c"])
    # Coulomb and drag blocks arrived in schema 3; older fits simply lack them.
    S = haskey(mb, "S") ?
        NTuple{9,Float32}(Float32(mb["S"][r][cc]) for r in 1:3 for cc in 1:3) :
        NTuple{9,Float32}(ntuple(_ -> 0.0f0, 9))
    D = haskey(mb, "D") ?
        NTuple{9,Float32}(Float32(mb["D"][r][cc]) for r in 1:3 for cc in 1:3) :
        NTuple{9,Float32}(ntuple(_ -> 0.0f0, 9))
    # A traction knee is required. The gains were fitted against the saturated
    # command, so a fit that arrives without one describes dynamics the solver
    # would then not reproduce -- it would overestimate what the robot can do
    # near full stick, which is exactly where the routes are planned. TOML
    # cannot express null, so an absent knee shows up as NaN here.
    haskey(mb, "traction_knee") && mb["traction_knee"] !== nothing ||
        error("regression has no traction_knee; refit with " *
              "fit_drivetrain.py --traction-knee auto")
    knee = Float32(mb["traction_knee"])
    (isnan(knee) || knee <= 0.0f0) &&
        error("regression traction_knee is $(mb["traction_knee"]); a " *
              "positive knee is required")
    eps = haskey(mb, "coulomb_eps") ?
        NTuple{3,Float32}(Float32(v) for v in mb["coulomb_eps"]) :
        (5.0f0, 5.0f0, 0.15f0)
    units = get(get(cfg, "units", Dict()), "distance", "cm")
    units == "cm" || @warn "regression distance unit is '$units', solver assumes cm"
    (Model(B, A, q, S, D, c, eps, knee), cfg)
end

function _poly(pts, what)
    length(pts) >= 3 || error("$what has fewer than 3 vertices")
    M = Matrix{Float64}(undef, 2, length(pts))
    for (i, p) in enumerate(pts)
        length(p) == 2 || error("$what has a vertex that is not [x, y]")
        M[1, i] = Float64(p[1]); M[2, i] = Float64(p[2])
    end
    M
end

"""
Load a field description: bounds, obstacle polygons, and the robot footprint.

Obstacles are the *real* physical obstacles, un-inflated. The robot polygon is
given in the robot's own body frame with the origin at the odometry tracking
point, and the solver works out the swept footprint per heading itself.

If `robot` is absent the robot is treated as a point, which is the old
behaviour and keeps existing field files working.
"""
function load_field(path::AbstractString)
    cfg = readjson(path)
    f = cfg.field
    bounds = (Float64(f.x_min), Float64(f.y_min), Float64(f.x_max), Float64(f.y_max))
    polys = Matrix{Float64}[]
    for ob in get(cfg, :obstacles, [])
        push!(polys, _poly(ob.polygon, "obstacle '$(get(ob, :name, "?"))'"))
    end
    robot = nothing
    if haskey(cfg, :robot) && haskey(cfg.robot, :polygon)
        robot = _poly(cfg.robot.polygon, "robot footprint")
    end
    (bounds, polys, robot, cfg)
end

"""Load the list of 6-vector target states."""
function load_targets(path::AbstractString)
    cfg = readjson(path)
    names = String[]
    states = NTuple{6,Float32}[]
    for t in cfg.targets
        s = t.state
        length(s) == 6 || error("target '$(get(t,:name,"?"))' state must be 6 numbers")
        push!(names, String(get(t, :name, "target$(length(names))")))
        push!(states, NTuple{6,Float32}(Float32(v) for v in s))
    end
    isempty(states) && error("$path lists no targets")
    (names, states, cfg)
end

# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

"""
Find the grid cells that count as "arrived".

The online optimizer decides what arrival really means, so this only has to
plant a seed for the recursion: the cells within `tol` of the target on every
axis, and always at least the single nearest cell so the seed can never be
empty.
"""
function target_cells(g::Grid6, s::NTuple{6,Float32}, tol::NTuple{6,Float32})
    idxs = Int64[]
    ranges = ntuple(6) do k
        c = (s[k] - g.lo[k]) / g.step[k]
        if k == 3
            n3 = Float32(g.n[3])
            c = c - floor(c / n3) * n3
        end
        r = tol[k] / g.step[k]
        # Strictly the cells whose centre lies within tol. Using floor/ceil
        # the other way round would widen the seed by up to a cell on each
        # side, and a seed that is too big reports "already arrived" for
        # states that genuinely need time -- badly misleading the robot.
        lo = max(0, ceil(Int, c - r))
        hi = min(Int(g.n[k]) - 1, floor(Int, c + r))
        k == 3 ? (ceil(Int, c - r):floor(Int, c + r)) : (lo:hi)
    end
    for i1 in ranges[1], i2 in ranges[2], i3 in ranges[3],
        i4 in ranges[4], i5 in ranges[5], i6 in ranges[6]
        j3 = mod(i3, Int(g.n[3]))
        push!(idxs, flatten(g, i1, i2, j3, i4, i5, i6))
    end
    if isempty(idxs)
        near = ntuple(6) do k
            c = (s[k] - g.lo[k]) / g.step[k]
            k == 3 ? mod(round(Int, c), Int(g.n[3])) :
                     clamp(round(Int, c), 0, Int(g.n[k]) - 1)
        end
        push!(idxs, flatten(g, near...))
    end
    unique!(idxs)
    idxs
end

# --------------------------------------------------------------------------
# Solve drivers
# --------------------------------------------------------------------------

"""
Asynchronous (in-place) value iteration.

Updating V in place rather than into a second buffer halves the memory -- the
difference between fitting a large grid in 8 GB of VRAM and not -- and also
converges faster, because improvements propagate within a sweep instead of
waiting for the next one. The races this introduces are benign: every write
only ever lowers a cell, so the iteration stays monotone and still converges
to the same fixed point. This is standard asynchronous value iteration.
"""
function sweep_cpu!(V, occ, pol, g, m, ctl, nctl, p::Params, phase::Integer;
                    rev::Bool = false)
    n = ncells(g)
    total = Threads.Atomic{Float64}(0.0)
    Threads.@threads for i in 1:n
        idx = rev ? Int64(n) - Int64(i) : Int64(i) - 1
        c, u1, u2, u3 = cell_update(idx, V, occ, pol, g, m, ctl, Int32(nctl),
                                    p, Int32(phase))
        @inbounds old = V[idx + 1]
        if c < old
            @inbounds V[idx + 1] = c
            pol === nothing || pol_set!(pol, idx, Int64(n), u1, u2, u3)
            Threads.atomic_add!(total, Float64(old - c))
        end
    end
    total[]
end

# --------------------------------------------------------------------------
# Encoding and output
# --------------------------------------------------------------------------

const DTYPES = Dict(
    "u8"  => (bytes = 1, sentinel = 0xff,       scale = 0.025),
    "u16" => (bytes = 2, sentinel = 0xffff,     scale = 0.001),
    "f16" => (bytes = 2, sentinel = NaN,        scale = 1.0),
    "f32" => (bytes = 4, sentinel = NaN,        scale = 1.0),
)

"""
Encode seconds into the chosen on-card representation.

A cell still sitting at `cap` was never reached, and becomes the dtype's
unreachable sentinel. Anything that saturates the integer range becomes the
sentinel too -- from the robot's point of view "longer than this table can
express" and "no route" are the same instruction: don't go there.
"""
function encode(V::AbstractVector{Float32}, dtype::String, scale::Float64,
                cap::Float32)
    buf = Vector{UInt8}(undef, length(V) * DTYPES[dtype].bytes)
    encode_into!(buf, V, Int64(firstindex(V)), Int64(lastindex(V)), dtype,
                 scale, cap)
    buf
end

"""
Encode `V[lo:hi]` into the front of `buf`, and count how many of those cells
were reached.

The range form is what lets a table be written without ever holding it, or
its encoding, in memory at once: a full-scale grid is tens of gigabytes per
target, so `encode` allocating a second copy of it is not a detail. The
counting rides along because the alternative is a second pass over the same
tens of gigabytes to learn one number.
"""
function encode_into!(buf::Vector{UInt8}, V::AbstractVector{Float32},
                      lo::Int64, hi::Int64, dtype::String, scale::Float64,
                      cap::Float32)
    unreached(v) = v >= cap || isinf(v) || isnan(v)
    n = hi - lo + 1
    reached = 0
    if dtype == "f32"
        out = reinterpret(Float32, view(buf, 1:4n))
        @inbounds for i in 1:n
            v = V[lo + i - 1]
            u = unreached(v)
            reached += !u
            out[i] = u ? NaN32 : v
        end
    elseif dtype == "f16"
        out = reinterpret(Float16, view(buf, 1:2n))
        @inbounds for i in 1:n
            v = V[lo + i - 1]
            u = unreached(v)
            reached += !u
            out[i] = u ? Float16(NaN) : Float16(v)
        end
    elseif dtype == "u16"
        out = reinterpret(UInt16, view(buf, 1:2n))
        @inbounds for i in 1:n
            v = V[lo + i - 1]
            u = unreached(v)
            reached += !u
            r = u ? 0xffff : round(UInt32, v / scale)
            out[i] = r >= 0xffff ? 0xffff : UInt16(r)
        end
    elseif dtype == "u8"
        @inbounds for i in 1:n
            v = V[lo + i - 1]
            u = unreached(v)
            reached += !u
            r = u ? 0xff : round(UInt32, v / scale)
            buf[i] = r >= 0xff ? 0xff : UInt8(r)
        end
    else
        error("unknown dtype '$dtype'")
    end
    reached
end

"""
Write one target's table as chunked 8.3-named files.

`chunk_elements` is a power of two so the robot resolves a cell with a shift
and a mask rather than a division. See docs/TABLE_FORMAT.md.
"""
function write_table(dir::AbstractString, target_index::Int, V,
                     dtype::String, scale::Float64, chunk_elements::Int,
                     cap::Float32)
    eb = DTYPES[dtype].bytes
    mkpath(dir)
    cells = table_cells(V)
    nchunks = Int(cld(cells, Int64(chunk_elements)))
    # One chunk of values, and one chunk of encoded bytes, allocated once and
    # refilled. The chunk is the unit the card is written in anyway, so
    # streaming through it costs nothing and bounds what this needs in memory
    # at a few tens of megabytes however large the table is -- which at full
    # scale is the difference between writing the card and not.
    vals = Vector{Float32}(undef, chunk_elements)
    buf = Vector{UInt8}(undef, chunk_elements * eb)
    ctx = SHA.SHA256_CTX()
    reached = Int64(0)
    total = Int64(0)
    for ci in 0:nchunks-1
        lo = Int64(ci) * Int64(chunk_elements) + 1
        hi = min(cells, Int64(ci + 1) * Int64(chunk_elements))
        n = hi - lo + 1
        table_chunk!(vals, V, lo, n)
        nb = Int(n * eb)
        reached += encode_into!(buf, vals, Int64(1), n, dtype, scale, cap)
        part = view(buf, 1:nb)
        name = @sprintf("T%02dC%04d.BIN", target_index, ci)
        open(joinpath(dir, name), "w") do io
            write(io, part)
        end
        SHA.update!(ctx, part)
        total += nb
    end
    (nchunks = nchunks, bytes = total, sha256 = bytes2hex(SHA.digest!(ctx)),
     reached = reached, cells = cells)
end

"""
How many values a table has, and how to get one chunk of them.

Two implementations, because a table is written either straight out of an
array -- device or host, whole grid in memory -- or out of the out-of-core
store, which is a file. `write_table` is written once against these so that
the two paths cannot drift apart in the chunking, the hashing or the
unreachable count.
"""
table_cells(V::AbstractVector{Float32}) = Int64(length(V))

table_chunk!(buf::Vector{Float32}, V::AbstractVector{Float32}, lo::Int64,
             n::Int64) = (copyto!(buf, 1, V, Int(lo), Int(n)); nothing)
