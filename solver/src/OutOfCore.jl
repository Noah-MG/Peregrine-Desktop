# Out-of-core value iteration: the grid lives on disk, one tile at a time
# lives on the GPU. Included into the PeregrineSolver module.
#
# The whole scheme rests on one property of the Bellman backup, argued in
# `Tiles.jl` and measured by `reach_extent`: a backup reads only within a
# bounded distance of the cell it updates. Load a window of the grid wide
# enough to contain that distance, update the middle of it, write the middle
# back, move on. Nothing about the backup itself changes -- `cell_update` is
# handed the tile's own `Grid6` and cannot tell the difference.
#
# That this converges to the same fixed point as the in-core solve is the
# same argument the in-core solve already relies on. Value iteration with
# in-place updates is asynchronous dynamic programming: the order cells are
# updated in does not matter, only that every cell keeps being updated. V
# starts at `cap`, every write lowers a cell, and every candidate evaluated
# is a real Bellman evaluation, so the iterate stays an upper bound on the
# true value and descends monotonically to the fixed point. Tiling is a
# reordering. Freezing a tile's halo for the length of its residency is a
# reordering. Neither is a new approximation.

# --------------------------------------------------------------------------
# The stores
# --------------------------------------------------------------------------

"""
A scratch array too big for RAM, kept in a file and read a slab at a time.

**Explicit reads and writes rather than a memory map**, which is what this
started as and had to stop being. On Windows a mapped file cannot be deleted
while its mapping is open, and the mapping is not released by finalising the
array, by closing the stream, or by any amount of garbage collection -- all
measured, all failing; only process exit frees it. A solve would therefore
leave its scratch behind every time, and at full scale that is tens of
gigabytes a run quietly accumulating on the volume the user pointed at.

Doing the I/O by hand costs one host buffer per resident plane and buys back
more than tidiness. A 32 GB dirty mapping on a 32 GB machine leaves the
write-back schedule to the operating system's judgement at exactly the moment
there is no memory spare for it to exercise judgement with; explicit reads
and writes of a few hundred megabytes at a time behave the same on the
thousandth round as on the first.

`path = nothing` keeps the whole thing in RAM instead, which is what the
self-test and small grids use. Every accessor below has both paths, so the
driver above cannot tell which it was given.
"""
struct ValueStore
    io::Union{Nothing,IOStream}
    ram::Union{Nothing,Vector{Float32}}
    path::Union{Nothing,String}
    cells::Int64
end

"""
The warm-start policy: one signed byte per command component, per cell.

Why bytes, and why interleaved per cell rather than in three planes, is
argued at `pol_get`. The device reads that layout directly, so a tile's
policy costs three resident bytes a cell rather than twelve and there is no
pack or unpack step at either end.

Kept across rounds and addressed by global cell, so the tiling moving from
one round to the next does not disturb it.
"""
struct PolicyStore
    io::Union{Nothing,IOStream}
    ram::Union{Nothing,Vector{Int8}}
    path::Union{Nothing,String}
    cells::Int64
end

const AnyStore = Union{ValueStore,PolicyStore}

"""
Create a file of exactly `nbytes` and leave it open for reading and writing.

Sized up front rather than grown. A file that runs out of room half way
through a solve fails at whichever tile happens to touch the end of it, hours
in; failing here instead costs one `seek` and a byte.
"""
function _new_scratch(path::AbstractString, nbytes::Int64)
    mkpath(dirname(path))
    io = open(path, "w+")
    if nbytes > 0
        seek(io, nbytes - 1)
        write(io, UInt8(0))
        flush(io)
    end
    io
end

function open_store(cells::Integer; path = nothing)
    n = Int64(cells)
    path === nothing &&
        return ValueStore(nothing, Vector{Float32}(undef, n), nothing, n)
    ValueStore(_new_scratch(path, n * sizeof(Float32)), nothing, String(path), n)
end

function open_policy_store(cells::Integer; path = nothing)
    n = Int64(cells)
    path === nothing &&
        return PolicyStore(nothing, zeros(Int8, 3 * n), nothing, n)
    PolicyStore(_new_scratch(path, 3 * n), nothing, String(path), n)
end

"""
A second, read-only handle on the same scratch file.

The prefetch reads a tile while the main loop is still writing the previous
one back, and `read_range!`/`write_range!` both `seek` the stream they are
given. Sharing one `IOStream` between them would interleave those seeks and
read from wherever the other one left the position -- silent, and not
reproducible. Separate handles have separate positions, so the two cannot
disturb each other.

A RAM-backed store has no handle to duplicate and is returned unchanged; it
exists for the self-test and small grids, where the prefetch is off anyway.
"""
read_handle(s::ValueStore) = s.path === nothing ? s :
    ValueStore(open(s.path, "r"), nothing, s.path, s.cells)
read_handle(s::PolicyStore) = s.path === nothing ? s :
    PolicyStore(open(s.path, "r"), nothing, s.path, s.cells)

"""Close a handle made by `read_handle`, leaving the original alone."""
close_handle!(h::AnyStore, orig::AnyStore) =
    (h !== orig && h.io !== nothing && close(h.io); nothing)

_elt(::ValueStore) = Float32
_elt(::PolicyStore) = Int8
_nelem(s::ValueStore) = s.cells
_nelem(s::PolicyStore) = 3 * s.cells

"""
Bounds check on every store access.

Deliberately not an `@inbounds`-style luxury. The offsets are six-dimensional
index arithmetic computed per tile per round, the reads go through raw
pointers, and a write past the end of the file would silently extend it
rather than fail -- so an arithmetic slip would show up as a wrong table
after a run of days, if at all. One comparison per plane transfer, against
transfers of hundreds of megabytes, costs nothing measurable.
"""
@inline function _in_store(s::AnyStore, off::Integer, n::Integer, bo::Integer,
                           buflen::Integer)
    lo = Int64(off); cnt = Int64(n); total = _nelem(s)
    (lo >= 0 && cnt >= 0 && lo + cnt <= total) ||
        error("store access [$lo, $(lo + cnt)) is outside its $total elements")
    (Int64(bo) >= 1 && Int64(bo) + cnt - 1 <= Int64(buflen)) ||
        error("store access needs buffer[$bo .. $(Int64(bo) + cnt - 1)] of $buflen")
    nothing
end

"""
Read `n` elements from 0-based element offset `off` into `buf` at 1-based
position `bo`.

`unsafe_read` on a raw pointer rather than `read!` on a view: `read!` falls
back to an element-at-a-time loop for anything that is not a dense `Array`,
and these are millions of elements at a time.
"""
function read_range!(buf::Vector{T}, bo::Integer, s::AnyStore, off::Integer,
                     n::Integer) where {T}
    n <= 0 && return nothing
    _in_store(s, off, n, bo, length(buf))
    if s.ram !== nothing
        copyto!(buf, Int(bo), s.ram, Int(off) + 1, Int(n))
    else
        seek(s.io, Int64(off) * sizeof(T))
        GC.@preserve buf unsafe_read(s.io, pointer(buf, bo), Int64(n) * sizeof(T))
    end
    nothing
end

"""Write `n` elements from `buf[bo:]` to 0-based element offset `off`."""
function write_range!(s::AnyStore, off::Integer, buf::Vector{T}, bo::Integer,
                      n::Integer) where {T}
    n <= 0 && return nothing
    _in_store(s, off, n, bo, length(buf))
    if s.ram !== nothing
        copyto!(s.ram, Int(off) + 1, buf, Int(bo), Int(n))
    else
        seek(s.io, Int64(off) * sizeof(T))
        GC.@preserve buf unsafe_write(s.io, pointer(buf, bo), Int64(n) * sizeof(T))
    end
    nothing
end

"""Set every element, in blocks, without ever holding the whole thing."""
function fill_store!(s::AnyStore, value)
    v = convert(_elt(s), value)
    total = _nelem(s)
    if s.ram !== nothing
        fill!(s.ram, v)
        return nothing
    end
    block = Int(min(total, Int64(1) << 22))
    buf = fill(v, block)
    off = Int64(0)
    while off < total
        n = min(Int64(block), total - off)
        write_range!(s, off, buf, 1, n)
        off += n
    end
    flush(s.io)
    nothing
end

"""Set individual cells, by 1-based flat index. This is the target seed."""
function set_cells!(s::ValueStore, idx::Vector{Int64}, value::Float32)
    buf = [value]
    for i in idx
        write_range!(s, i - 1, buf, 1, 1)
    end
    nothing
end

# The out-of-core half of the table-writing interface; see `table_cells`.
table_cells(s::ValueStore) = s.cells
table_chunk!(buf::Vector{Float32}, s::ValueStore, lo::Int64, n::Int64) =
    read_range!(buf, 1, s, lo - 1, n)

"""The whole store as an ordinary vector. For tests and small grids only."""
function read_all(s::ValueStore)
    s.ram === nothing || return copy(s.ram)
    out = Vector{Float32}(undef, s.cells)
    read_range!(out, 1, s, 0, s.cells)
    out
end

"""
Close a store and drop its scratch file unless asked to keep it.

A delete that fails anyway is reported rather than swallowed. It does not
make the tables wrong, so it must not fail a finished run; but a file this
size going quietly missing from the disk budget is exactly what nobody
notices until the volume is full.
"""
function close_store!(s::AnyStore; keep::Bool = false)
    s.io === nothing && return nothing
    flush(s.io)
    close(s.io)
    (keep || s.path === nothing || !isfile(s.path)) && return nothing
    try
        rm(s.path)
    catch e
        @warn("could not delete the solver scratch file; remove it by hand",
              path = s.path, bytes = filesize(s.path), exception = e)
    end
    nothing
end

# --------------------------------------------------------------------------
# Moving a tile between a store and the device
# --------------------------------------------------------------------------

"""
Copy one tile's window out of a store and onto the device, a plane at a time.

The unit is an x plane, not a column. In the row-major index `x` is slowest
and `y` next, so a whole y span at fixed `x` is contiguous in *both* the store
and the tile, and a tile takes a contiguous y span by construction. Doing it
column by column would move the same bytes in a few thousand times as many
reads.

Staging one plane rather than the whole window is what keeps the host side
small: at full scale a window runs to several gigabytes while a plane is a
few hundred megabytes, and the extra device copies are free next to the read.

`per` is elements per cell -- one for `V`, three for the policy.
"""
function load_tile!(dst, s::AnyStore, g::Grid6, t::TileSpec, col::Int64,
                    buf::Vector{T}, per::Int64 = Int64(1)) where {T}
    n2 = Int64(g.n[2])
    run = Int64(t.nyl) * col * per
    for jx in 0:(t.nxl - 1)
        src = ((Int64(t.lx0 + jx) * n2) + Int64(t.ly0)) * col * per
        read_range!(buf, 1, s, src, run)
        copyto!(dst, Int64(jx) * run + 1, buf, 1, run)
    end
    nothing
end

"""
Read one tile's window into a host buffer, in the layout the device wants.

The same plane walk as `load_tile!`, minus the device: `read_range!` puts
each plane straight at its place in `win`, so a later `copyto!` of the whole
buffer reproduces exactly what `load_tile!` would have built. That is the
whole point -- it lets the read happen on a background task, with every CUDA
call left on the task that owns the sweeps.

Returns the number of elements written, since an edge tile fills only a
prefix of a buffer sized for the worst case.
"""
function read_window!(win::Vector{T}, s::AnyStore, g::Grid6, t::TileSpec,
                      col::Int64, per::Int64 = Int64(1)) where {T}
    n2 = Int64(g.n[2])
    run = Int64(t.nyl) * col * per
    for jx in 0:(t.nxl - 1)
        src = ((Int64(t.lx0 + jx) * n2) + Int64(t.ly0)) * col * per
        read_range!(win, Int64(jx) * run + 1, s, src, run)
    end
    Int64(t.nxl) * run
end

"""
Copy a tile's interior back to a store. The halo is deliberately not written:
those cells belong to other tiles, and this pass held them frozen.
"""
function store_tile!(s::AnyStore, src, g::Grid6, t::TileSpec, col::Int64,
                     buf::Vector{T}, per::Int64 = Int64(1)) where {T}
    n2 = Int64(g.n[2])
    ox = Int64(t.ix0 - t.lx0); oy = Int64(t.iy0 - t.ly0)
    run = Int64(t.wy) * col * per
    for jx in 0:(t.wx - 1)
        soff = ((ox + Int64(jx)) * Int64(t.nyl) + oy) * col * per + 1
        copyto!(buf, 1, src, soff, run)
        doff = ((Int64(t.ix0 + jx) * n2) + Int64(t.iy0)) * col * per
        write_range!(s, doff, buf, 1, run)
    end
    nothing
end

"""
Slice the occupancy mask to the tile's window.

Occupancy is indexed `(x*Ny + y)*Nh + h`, the same nesting as the value grid
with a column of `Nh` instead of `Nrest`, so the same plane-at-a-time copy
works on it. It is small enough to stay in RAM whole, so this reads from the
array rather than from a store.
"""
function tile_occupancy!(host::Vector{Bool}, occ::Vector{Bool}, g::Grid6,
                         t::TileSpec)
    n2 = Int64(g.n[2]); n3 = Int64(g.n[3])
    run = Int64(t.nyl) * n3
    for jx in 0:(t.nxl - 1)
        src = ((Int64(t.lx0 + jx) * n2) + Int64(t.ly0)) * n3 + 1
        dst = Int64(jx) * run + 1
        copyto!(host, dst, occ, src, run)
    end
    Int64(t.nxl) * run
end

"""
The tile-local indices of the seed cells this tile is responsible for.

Only interior seeds are returned. A seed sitting in the halo is already zero
in the store, put there when its own tile wrote it, and re-asserting it here
would be writing to a cell this pass does not own.
"""
function tile_seeds(tidx::Vector{Int64}, g::Grid6, t::TileSpec, col::Int64)
    n2 = Int64(g.n[2])
    plane = n2 * col
    out = Int64[]
    for gi in tidx
        i = gi - 1                       # tidx is 1-based
        gx = i ÷ plane; rem = i % plane
        gy = rem ÷ col;  r = rem % col
        (t.ix0 <= gx < t.ix0 + t.wx) || continue
        (t.iy0 <= gy < t.iy0 + t.wy) || continue
        jx = gx - Int64(t.lx0); jy = gy - Int64(t.ly0)
        push!(out, (jx * Int64(t.nyl) + jy) * col + r + 1)
    end
    out
end

# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------

"""
Out-of-core value iteration for one target.

Mirrors `solve_value!` -- same seeding, same monotone descent, same stopping
rule -- with the sweep split into tiles and each tile swept `tile_sweeps`
times while it is resident.

**Why more than one sweep per residency.** Loading a tile costs its whole
window; sweeping it costs only its interior. The window is larger than the
interior by the halo, and on a grid where the halo is comparable to the tile
that ratio is the entire cost of the scheme. Sweeping `k` times before
writing back divides it by `k`. The sweeps after the first run against a
frozen halo, which is block Gauss-Seidel: still a valid backup, still
monotone, and in practice faster per unit of work than the synchronous
version, because information crosses the tile immediately instead of waiting
for the next round.

**What the shifting tiling buys.** Moving the tile boundaries every round was
put in to stop the frozen-halo bias settling into a fixed lattice of seams,
and it turns out to do more than that: it makes the scheme forgiving of a
halo that is too small. A cell near a tile edge loses the candidate steps
that would leave the window, but next round the boundaries have moved and it
is somewhere in the middle with its whole reach inside -- and because V only
ever decreases, the better answer it finds then is the one that is kept.
Measured: on a grid whose reach was a third of a tile, running with *no halo
at all* came out level with the honest halo, because the shift healed it.

That is a safety net, not a licence. It only works while the reach is small
next to the tile, which is the comfortable case and not the one a full-scale
run is in -- there the tile is barely wider than the halo, no shift can put a
cell clear of every boundary, and the halo is load bearing. `shift = false`
is what the self-test uses to see the difference.

Convergence is judged per round -- one full pass over every tile -- rather
than per sweep, because that is the unit in which every cell has been
updated. The quiet-round requirement folds in the rotating control slice for
the same reason `solve_value!` folds it into the quiet-sweep requirement: a
quiet round proves only that the controls this round happened to scan do not
help, so the run has to stay quiet long enough for the whole lattice to have
had its turn.
"""
function solve_value_ooc!(s::ValueStore, occ::Vector{Bool}, g::Grid6, m::Model,
                          tidx::Vector{Int64}, ctl_h::Vector{Float32}, nctl,
                          p::Params, tp::TilePlan;
                          rounds::Int, tol::Float64, tile_sweeps::Int = 4,
                          warm::Bool = false,
                          pstore::Union{Nothing,PolicyStore} = nothing,
                          shift::Bool = true, prefetch::Bool = false,
                          on_progress = nothing, on_tile = nothing)
    col = tp.col
    fill_store!(s, p.cap)
    set_cells!(s, tidx, 0.0f0)
    pstore === nothing || fill_store!(pstore, Int8(0))

    # Device buffers, sized for the worst-case tile and reused by every tile
    # of every round. The kernel never addresses past the current tile's own
    # cell count, so a smaller edge tile simply uses a prefix.
    maxcells = Int64(tp.nxl) * Int64(tp.nyl) * col
    dV = CUDA.zeros(Float32, maxcells)
    # Bytes, not floats: `pol_get`/`pol_set!` read this layout directly, so
    # there is no unpacking step and the policy costs 3 bytes a resident cell
    # rather than 12. At full scale that is the difference between warm
    # starting and not being able to afford to.
    dpol = warm ? CUDA.zeros(Int8, 3 * maxcells) : nothing
    docc = CUDA.zeros(Bool, Int64(tp.nxl) * Int64(tp.nyl) * Int64(g.n[3]))
    hocc = Vector{Bool}(undef, Int64(tp.nxl) * Int64(tp.nyl) * Int64(g.n[3]))
    dctl = CuArray(ctl_h)
    dtotal = CUDA.zeros(Float32, 1)
    # Host staging: one x plane of the widest window, which is what bounds
    # what this needs in RAM however large the grid gets.
    plane = Int64(tp.nyl) * col
    hV = Vector{Float32}(undef, plane)
    hP = warm ? Vector{Int8}(undef, 3 * plane) : Int8[]

    # Prefetch. The load of a tile is pure host work -- seeks and reads on the
    # scratch -- and the sweeps that follow it are a GPU kernel the host only
    # waits on, so the two can run at once and the read costs nothing. What
    # this buys is bounded by the read: it turns a round from compute + io
    # into max(compute, read) + write, and it is therefore worth most exactly
    # where the card is fastest.
    #
    # Two things make it safe rather than merely fast.
    #
    # **Separate file handles.** `read_handle` gives the background task its
    # own position, so its seeks and the write-back's cannot interleave.
    #
    # **A stale halo is still a valid backup.** Tile k+1 is read while tile k
    # is still being written, so where their windows overlap the prefetch can
    # see the older value. That is sound, and it is sound for the same reason
    # the whole scheme is: V only ever decreases, so an older V is a larger V,
    # and `cell_update` takes a min against it. A stale read can therefore
    # make a cell converge later; it can never make it converge to something
    # too small, which is the direction that would be dangerous. It is the
    # same benign race the in-core driver already runs on purpose.
    #
    # Not carried across a round boundary: the next round's tiling depends on
    # the shift, and the loop may stop at the convergence check, so the last
    # tile of each round loads the ordinary way.
    pre_s = prefetch ? read_handle(s) : s
    pre_ps = (prefetch && pstore !== nothing) ? read_handle(pstore) : pstore
    winV = prefetch ? Vector{Float32}(undef, maxcells) : Float32[]
    winP = (prefetch && warm && pstore !== nothing) ?
           Vector{Int8}(undef, 3 * maxcells) : Int8[]
    pretask = nothing
    pren = Int64(0)

    quiet_needed = p.ncoarse <= Int32(0) ? 1 :
        max(1, cld(Int(nctl) - 1, Int(p.ncoarse) * max(tile_sweeps, 1)))

    phase = 0
    quiet = 0
    last_delta = Inf
    done_rounds = 0

    for rd in 1:rounds
        tiles = tiles_of(g, tp; shift = shift ? rd - 1 : 0)
        delta = 0.0
        for (k, t) in enumerate(tiles)
            gt = subgrid(g, t.lx0, t.ly0, t.nxl, t.nyl)
            # Either the background task already has this window in host RAM,
            # or nobody fetched it and it is read the ordinary way.
            if pretask !== nothing
                wait(pretask)
                pretask = nothing
                copyto!(dV, 1, winV, 1, pren)
                dpol === nothing || pstore === nothing ||
                    copyto!(dpol, 1, winP, 1, 3 * pren)
            else
                load_tile!(dV, s, g, t, col, hV)
                dpol === nothing || pstore === nothing ||
                    load_tile!(dpol, pstore, g, t, col, hP, Int64(3))
            end
            nocc = tile_occupancy!(hocc, occ, g, t)
            copyto!(docc, 1, hocc, 1, nocc)
            # The policy either travels with the tile from its own store --
            # addressed by global cell, so the shifting tile boundaries do not
            # disturb it -- or, with no store behind it, lasts only as long as
            # this residency and starts empty.
            dpol === nothing || pstore !== nothing || fill!(dpol, Int8(0))

            # Start the next window before the sweeps, not after: the sweeps
            # are what there is to hide it behind. `sweep_tile_gpu!` waits on
            # the device with a yielding synchronise, so the scheduler is free
            # to run this task while the kernel is in flight.
            if prefetch && k < length(tiles)
                tn = tiles[k + 1]
                pretask = Threads.@spawn begin
                    n = read_window!(winV, pre_s, g, tn, col)
                    if !isempty(winP)
                        read_window!(winP, pre_ps, g, tn, col, Int64(3))
                    end
                    n
                end
                pren = Int64(tn.nxl) * Int64(tn.nyl) * col
            end
            seeds = tile_seeds(tidx, g, t, col)
            dseeds = isempty(seeds) ? nothing : CuArray(seeds)
            ox = Int64(t.ix0 - t.lx0); oy = Int64(t.iy0 - t.ly0)

            for _ in 1:tile_sweeps
                phase += 1
                delta += sweep_tile_gpu!(dV, docc, dpol, gt, m, dctl, nctl, p,
                                         phase, dtotal, ox, oy, t.wx, t.wy;
                                         rev = isodd(phase))
                # The seed must be reasserted, for the same reason it is
                # in-core: interpolation noise can otherwise walk a target
                # cell below zero and corrupt its whole basin.
                dseeds === nothing || (@views dV[dseeds] .= 0.0f0)
            end

            store_tile!(s, dV, g, t, col, hV)
            dpol === nothing || pstore === nothing ||
                store_tile!(pstore, dpol, g, t, col, hP, Int64(3))
            # Freed rather than left to the collector: a full-scale run makes
            # one of these per tile per round, which is hundreds of thousands
            # of small device allocations over a solve.
            dseeds === nothing || CUDA.unsafe_free!(dseeds)
            on_tile === nothing || on_tile(rd, k, length(tiles))
        end

        done_rounds = rd
        # Reported per sweep, not per round, so `tolerance` means the same
        # thing here as it does in core. A round is `tile_sweeps` sweeps of
        # every cell, so its total improvement is that many times larger for
        # the same rate of progress -- comparing it against the in-core
        # tolerance would just stop later for no reason.
        last_delta = delta / max(tile_sweeps, 1)
        quiet = last_delta <= tol ? quiet + 1 : 0
        # Every round. A round at full scale is minutes to tens of minutes,
        # so there is nothing to be saved by reporting every other one, and a
        # bar that only moves half as often is a bar the user distrusts.
        if on_progress !== nothing
            on_progress(rd, last_delta)
        end
        quiet >= quiet_needed && break
    end

    # A round never leaves one outstanding -- the last tile of a round is not
    # prefetched -- but an exception thrown mid-round could, and a task still
    # reading a handle that is about to be closed is a crash on top of
    # whatever actually went wrong.
    pretask === nothing || wait(pretask)
    close_handle!(pre_s, s)
    pstore === nothing || close_handle!(pre_ps, pstore)

    s.io === nothing || flush(s.io)
    pstore === nothing || pstore.io === nothing || flush(pstore.io)
    CUDA.unsafe_free!(dV)
    dpol === nothing || CUDA.unsafe_free!(dpol)
    CUDA.unsafe_free!(docc)
    CUDA.unsafe_free!(dctl)
    CUDA.unsafe_free!(dtotal)
    (done_rounds, last_delta)
end

"""
The escape pass, tiled, against a store that already holds a solved table.

`solve_escape!` for a grid too big for the card. Same tiling, same shift, same
round accounting as `solve_value_ooc!`; the differences are all consequences
of running *after* a solve rather than instead of one:

  * **The store is not filled and nothing is seeded.** The converged table is
    the initial condition -- see `solve_escape!`.
  * **No policy.** The warm start is worth its second store over hundreds of
    rounds of a full solve; this is tens of rounds over a fraction of the
    cells, and skipping it leaves `pstore` holding the real policy untouched.
  * **The improvement is measured in escape time**, since that is the
    quantity `sweep_tile_escape_gpu!` minimises.

**The stale-halo argument survives, read the other way up.** The prefetch is
sound in `solve_value_ooc!` because V only ever decreases, so an older halo
value is a larger one and `cell_update` mins against it. Here the stored value
is the *negated* escape time, so it only ever increases -- but the quantity
being minimised is still the escape time, an older halo still carries a larger
one, and the backup still mins. The direction that would be dangerous, a stale
read making a cell converge to something too small, is closed either way.
"""
function solve_escape_ooc!(s::ValueStore, occ::Vector{Bool}, g::Grid6,
                           m::Model, ctl_h::Vector{Float32}, nctl, p::Params,
                           tp::TilePlan;
                           rounds::Int, tol::Float64, tile_sweeps::Int = 4,
                           shift::Bool = true, prefetch::Bool = false,
                           on_progress = nothing, on_tile = nothing)
    col = tp.col
    maxcells = Int64(tp.nxl) * Int64(tp.nyl) * col
    dV = CUDA.zeros(Float32, maxcells)
    docc = CUDA.zeros(Bool, Int64(tp.nxl) * Int64(tp.nyl) * Int64(g.n[3]))
    hocc = Vector{Bool}(undef, Int64(tp.nxl) * Int64(tp.nyl) * Int64(g.n[3]))
    dctl = CuArray(ctl_h)
    dtotal = CUDA.zeros(Float32, 1)
    plane = Int64(tp.nyl) * col
    hV = Vector{Float32}(undef, plane)

    pre_s = prefetch ? read_handle(s) : s
    winV = prefetch ? Vector{Float32}(undef, maxcells) : Float32[]
    pretask = nothing
    pren = Int64(0)

    quiet_needed = p.ncoarse <= Int32(0) ? 1 :
        max(1, cld(Int(nctl) - 1, Int(p.ncoarse) * max(tile_sweeps, 1)))

    phase = 0
    quiet = 0
    last_delta = Inf
    done_rounds = 0

    for rd in 1:rounds
        tiles = tiles_of(g, tp; shift = shift ? rd - 1 : 0)
        delta = 0.0
        for (k, t) in enumerate(tiles)
            gt = subgrid(g, t.lx0, t.ly0, t.nxl, t.nyl)
            if pretask !== nothing
                wait(pretask)
                pretask = nothing
                copyto!(dV, 1, winV, 1, pren)
            else
                load_tile!(dV, s, g, t, col, hV)
            end
            nocc = tile_occupancy!(hocc, occ, g, t)
            copyto!(docc, 1, hocc, 1, nocc)

            if prefetch && k < length(tiles)
                tn = tiles[k + 1]
                pretask = Threads.@spawn read_window!(winV, pre_s, g, tn, col)
                pren = Int64(tn.nxl) * Int64(tn.nyl) * col
            end
            ox = Int64(t.ix0 - t.lx0); oy = Int64(t.iy0 - t.ly0)

            for _ in 1:tile_sweeps
                phase += 1
                delta += sweep_tile_escape_gpu!(dV, docc, gt, m, dctl, nctl, p,
                                                phase, dtotal, ox, oy,
                                                t.wx, t.wy; rev = isodd(phase))
            end

            store_tile!(s, dV, g, t, col, hV)
            on_tile === nothing || on_tile(rd, k, length(tiles))
        end

        done_rounds = rd
        last_delta = delta / max(tile_sweeps, 1)
        quiet = last_delta <= tol ? quiet + 1 : 0
        on_progress === nothing || on_progress(rd, last_delta)
        quiet >= quiet_needed && break
    end

    pretask === nothing || wait(pretask)
    close_handle!(pre_s, s)
    s.io === nothing || flush(s.io)
    CUDA.unsafe_free!(dV)
    CUDA.unsafe_free!(docc)
    CUDA.unsafe_free!(dctl)
    CUDA.unsafe_free!(dtotal)
    (done_rounds, last_delta)
end
