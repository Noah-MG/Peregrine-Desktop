# CUDA driver. Included into the PeregrineSolver module.

using CUDA

function gpu_sweep_kernel!(V, occ, pol, g::Grid6, m::Model, ctl, nctl::Int32,
                           p::Params, phase::Int32, rev::Bool, n::Int64, total)
    i = (Int64(blockIdx().x) - Int64(1)) * Int64(blockDim().x) + Int64(threadIdx().x)
    i > n && return nothing
    # Alternating sweep direction. Updates are in place, and thread blocks
    # retire in roughly increasing order, so information travels down the
    # index order much faster than up it. Reversing every other sweep gives
    # the other direction a turn, which costs nothing and removes the bias.
    idx = rev ? n - i : i - Int64(1)
    c, u1, u2, u3 = cell_update(idx, V, occ, pol, g, m, ctl, nctl, p, phase)
    @inbounds old = V[idx + 1]
    if c < old
        @inbounds V[idx + 1] = c
        pol === nothing || pol_set!(pol, idx, n, u1, u2, u3)
        # Total improvement this sweep: one scalar convergence signal, with no
        # second pass over the array.
        CUDA.@atomic total[1] += (old - c)
    end
    return nothing
end

"""One in-place sweep on the GPU. Returns the total improvement."""
function sweep_gpu!(V, occ, pol, g::Grid6, m::Model, ctl, nctl, p::Params,
                    phase::Integer, total; rev::Bool = false)
    n = ncells(g)
    fill!(total, 0.0f0)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks gpu_sweep_kernel!(
        V, occ, pol, g, m, ctl, Int32(nctl), p, Int32(phase), rev, n, total)
    CUDA.synchronize()
    Float64(CUDA.@allowscalar total[1])
end

"""
One in-place sweep over the *interior* of a resident tile.

`gt` is the tile's own grid (see `subgrid`), so every read and write in
`cell_update` is already tile-local and this kernel only has to hand it the
right index. The mapping is the whole difference between this and
`gpu_sweep_kernel!`: threads enumerate interior cells, which are a strided
subset of the loaded window, and the halo is updated by no one -- it is the
frozen boundary condition for this residency.

`nrest` is the cell count of one (x, y) column, so the interior decomposes as
(column-within-tile, cell-within-column) and consecutive threads stay
consecutive in memory, exactly as they do on the whole grid.
"""
function gpu_tile_sweep_kernel!(V, occ, pol, gt::Grid6, m::Model, ctl,
                                nctl::Int32, p::Params, phase::Int32, rev::Bool,
                                ox::Int64, oy::Int64, wx::Int64, wy::Int64,
                                nrest::Int64, nint::Int64, total)
    i = (Int64(blockIdx().x) - Int64(1)) * Int64(blockDim().x) + Int64(threadIdx().x)
    i > nint && return nothing
    j = rev ? nint - i : i - Int64(1)
    r  = j % nrest;  q  = j ÷ nrest
    jy = q % wy;     jx = q ÷ wy
    idx = ((jx + ox) * Int64(gt.n[2]) + (jy + oy)) * nrest + r
    c, u1, u2, u3 = cell_update(idx, V, occ, pol, gt, m, ctl, nctl, p, phase)
    @inbounds old = V[idx + 1]
    if c < old
        @inbounds V[idx + 1] = c
        pol === nothing || pol_set!(pol, idx, ncells(gt), u1, u2, u3)
        CUDA.@atomic total[1] += (old - c)
    end
    return nothing
end

"""One in-place sweep over a tile's interior. Returns the total improvement."""
function sweep_tile_gpu!(V, occ, pol, gt::Grid6, m::Model, ctl, nctl, p::Params,
                         phase::Integer, total, ox::Integer, oy::Integer,
                         wx::Integer, wy::Integer; rev::Bool = false)
    nrest = ncells(gt) ÷ (Int64(gt.n[1]) * Int64(gt.n[2]))
    nint = Int64(wx) * Int64(wy) * nrest
    fill!(total, 0.0f0)
    threads = 256
    blocks = cld(nint, threads)
    @cuda threads=threads blocks=blocks gpu_tile_sweep_kernel!(
        V, occ, pol, gt, m, ctl, Int32(nctl), p, Int32(phase), rev,
        Int64(ox), Int64(oy), Int64(wx), Int64(wy), nrest, nint, total)
    CUDA.synchronize()
    Float64(CUDA.@allowscalar total[1])
end

"""Device memory this run may use, in bytes: what is free, less headroom."""
function gpu_budget(; headroom = 0.85)
    CUDA.functional() || return Int64(0)
    free, _ = CUDA.memory_info()
    floor(Int64, free * headroom)
end

"""
Report whether this grid will fit whole, before anything is allocated.

V is 4 bytes per cell and the warm-start policy 3 more, so the requirement is
`cell_bytes(warm)` -- the occupancy mask is only Nx*Ny*Nh and the control set
is a handful of floats. Updating V in place rather than double-buffering is
what keeps the V term at one copy.

**Taken from `cell_bytes`, not written out again here.** This function used to
carry its own 16, which was right for the in-core policy of the day and wrong
the moment the tiled driver quantised its own; `decompose` and this function
then disagreed by 2.29x about the same grid.

Not fitting is no longer fatal: it selects the out-of-core driver, which
solves one tile of the grid at a time against a store on disk. See `Tiles.jl`
for what that costs and why the cut is where it is.
"""
function gpu_fits(g::Grid6; headroom = 0.85, warm::Bool = true)
    CUDA.functional() || return (false, 0.0, 0.0)
    need = ncells(g) * cell_bytes(warm) / 2^30
    free, _ = CUDA.memory_info()
    (need <= free / 2^30 * headroom, need, free / 2^30)
end
