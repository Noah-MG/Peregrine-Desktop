# CUDA driver. Included into the PeregrineSolver module.

using CUDA

function gpu_sweep_kernel!(V, occ, g::Grid6, m::Model, ctl, nctl::Int32,
                           dt::Float32, nsub::Int32, checks::Int32,
                           nearest::Bool, cap::Float32, n::Int64, total)
    i = (Int64(blockIdx().x) - Int64(1)) * Int64(blockDim().x) + Int64(threadIdx().x)
    i > n && return nothing
    idx = i - Int64(1)
    c = cell_update(idx, V, occ, g, m, ctl, nctl, dt, nsub, checks, nearest, cap)
    @inbounds old = V[i]
    if c < old
        @inbounds V[i] = c
        # Total improvement this sweep: one scalar convergence signal, with no
        # second pass over the array.
        CUDA.@atomic total[1] += (old - c)
    end
    return nothing
end

"""One in-place sweep on the GPU. Returns the total improvement."""
function sweep_gpu!(V, occ, g::Grid6, m::Model, ctl, nctl, dt, nsub, checks,
                    nearest, cap, total)
    n = ncells(g)
    fill!(total, 0.0f0)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks gpu_sweep_kernel!(
        V, occ, g, m, ctl, Int32(nctl), Float32(dt), Int32(nsub),
        Int32(checks), nearest, Float32(cap), n, total)
    CUDA.synchronize()
    Float64(CUDA.@allowscalar total[1])
end

"""
Report whether this grid will fit, before anything is allocated.

Only V lives on the device -- the occupancy mask is Nx*Ny and the control set
is a handful of floats -- so the requirement is essentially 4 bytes per cell.
That is the whole reason the solver updates in place rather than
double-buffering.
"""
function gpu_fits(g::Grid6; headroom = 0.85)
    CUDA.functional() || return (false, 0.0, 0.0)
    need = ncells(g) * 4 / 2^30
    free, _ = CUDA.memory_info()
    (need <= free / 2^30 * headroom, need, free / 2^30)
end
