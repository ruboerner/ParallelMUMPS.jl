module ParallelMUMPS

using Distributed
using MPI
using MUMPS
using SparseArrays
using LinearAlgebra

# BLAS.set_num_threads(1)

export init_workers!,
    factorize_shifts_grouped!,
    solve_block_all_xis, 
    solve_columns_all_xis,   
    free_factors!,
    finalize_workers!

# -----------------------------------------------------------------------------
# Worker-local state and helpers
# -----------------------------------------------------------------------------

const FACTORS = Dict{Int,Any}()

"""
    _init_local_worker!()

Initialize MPI on the current process if needed.
"""
function _init_local_worker!()
    if !MPI.Initialized()
        MPI.Init()
    end
    return nothing
end

"""
    _free_local_factors!()

Free cached MUMPS factors on the current process only.
"""
function _free_local_factors!()
    for m in values(FACTORS)
        finalize(m)
    end
    empty!(FACTORS)
    return nothing
end

"""
    _shutdown_local_worker!()

Free local factors and finalize MPI on the current process.
"""
function _shutdown_local_worker!()
    _free_local_factors!()
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
    end
    return nothing
end

"""
    _assemble_shift!(A, K, M, ξ)

Overwrite the numeric values of `A` with `K - ξ*M` without reallocating the
CSC structure. This requires `A`, `K`, and `M` to have identical sparsity
patterns.
"""
function _assemble_shift!(A::SparseMatrixCSC, K::SparseMatrixCSC, M::SparseMatrixCSC, ξ)
    (A.colptr == K.colptr == M.colptr && A.rowval == K.rowval == M.rowval) ||
        throw(ArgumentError("K, M, and A must have identical sparsity patterns"))

    Av = A.nzval
    Kv = K.nzval
    Mv = M.nzval
    @inbounds @simd for p in eachindex(Av)
        Av[p] = Kv[p] - ξ * Mv[p]
    end
    return A
end



"""
    build_factor!(i, A)

Build or rebuild the factorization for shift index `i` on the current process.
"""
function build_factor!(i, A)
    # A = K - ξ * M
    if haskey(FACTORS, i)
        finalize(FACTORS[i])
        delete!(FACTORS, i)
    end
    icntl = get_icntl(verbose=false)
    m = Mumps{eltype(A)}(mumps_unsymmetric, icntl, default_cntl64)
    associate_matrix!(m, A)
    factorize!(m)
    FACTORS[i] = m
    return nothing
end

"""
    build_factor!(i, K, M, ξ)

Build or rebuild the factorization for shift index `i` on the current process.
"""
function build_factor!(i, K, M, ξ)
    A = K - ξ * M
    if haskey(FACTORS, i)
        finalize(FACTORS[i])
        delete!(FACTORS, i)
    end
    icntl = get_icntl(verbose=false)
    m = Mumps{eltype(A)}(mumps_unsymmetric, icntl, default_cntl64)
    associate_matrix!(m, A)
    factorize!(m)
    FACTORS[i] = m
    return nothing
end


"""
    build_many_factors!(idxs, K, M, xis)

Build factors for several shift indices on the current process.
This reuses one sparse workspace matrix to avoid allocating `K - ξ*M`
for every shift. `K` and `M` must have identical sparsity patterns.

"""
function build_many_factors!(idxs, K::SparseMatrixCSC, M::SparseMatrixCSC, xis)
    length(idxs) == length(xis) || throw(ArgumentError("idxs and xis must have same length"))

        # K.colptr == M.colptr || throw(ArgumentError("K and M must have identical sparsity patterns"))
    # K.rowval == M.rowval || throw(ArgumentError("K and M must have identical sparsity patterns"))

    do_A = (K.colptr == M.colptr) & (K.rowval == M.rowval) 
    
    if do_A
        # Awork = copy(K)
        Twork = promote_type(eltype(K), eltype(M), eltype(xis))
        Awork = SparseMatrixCSC(
            size(K, 1),
            size(K, 2),
            copy(K.colptr),
            copy(K.rowval),
            Vector{Twork}(undef, nnz(K)),
        )
    end

    for k in eachindex(idxs)
        # build_factor!(idxs[k], (K - xis[k] * M) )
        if do_A 
            _assemble_shift!(Awork, K, M, xis[k])
            build_factor!(idxs[k], Awork)
        else
            build_factor!(idxs[k], K, M, xis[k])
        end
    end
    return nothing
end


"""
solve_same_rhs_with_factors(idxs, B)

Solve local systems for several shift indices with the same RHS block `B`.
    Returns solutions in the same order as `idxs`.
    """
function solve_same_rhs_with_factors(idxs, B)
    Xs = Vector{Any}(undef, length(idxs))
    for k in eachindex(idxs)
        m = FACTORS[idxs[k]]
        associate_rhs!(m, B)
        solve!(m)
        Xs[k] = get_solution(m)
        MUMPS.set_job!(m, 4)
    end
    return Xs
end

"""
    solve_matching_columns_with_factors(idxs, B)

Solve local systems for several shift indices where the right-hand side for
shift `idxs[k]` is the corresponding column `B[:, idxs[k]]`.

Returns a vector `xs` in the same order as `idxs`, where each entry is the
solution vector for the matching shift.
"""
function solve_matching_columns_with_factors(idxs, B)
    size(B, 2) >= maximum(idxs) || throw(ArgumentError("B must have at least one column per shift index"))

    xs = Vector{Any}(undef, length(idxs))
    for k in eachindex(idxs)
        i = idxs[k]
        m = FACTORS[i]
        bi = copy(B[:, i])
        associate_rhs!(m, bi)
        solve!(m)
        xs[k] = copy(get_solution(m))
        MUMPS.set_job!(m, 4)
    end
    return xs
end

# -----------------------------------------------------------------------------
# Distributed API
# -----------------------------------------------------------------------------

"""
    init_workers!()

Load required packages on all current workers and initialize MPI there.
Call `addprocs(...)` first.
"""
function init_workers!()
    ws = workers()
    isempty(ws) && error("No workers. Call addprocs(...) first.")

    @sync for w in ws
        @async remotecall_wait(_init_local_worker!, w)
    end

    return nothing
end

"""
    factorize_shifts_grouped!(owner, K, M, xis)

Factorize all shifted systems `K - ξ_i M` in parallel, grouped by owning worker.

`owner[i]` must be the worker id responsible for shift `i`.
"""
function factorize_shifts_grouped!(owner::Dict{Int,Int}, K, M, xis)
    ws = unique(values(owner))

    grouped_idxs = Dict(w => Int[] for w in ws)
    grouped_xis = Dict(w => eltype(xis)[] for w in ws)

    for i in eachindex(xis)
        w = owner[i]
        push!(grouped_idxs[w], i)
        push!(grouped_xis[w], xis[i])
    end

    @sync for w in ws
        isempty(grouped_idxs[w]) && continue
        @async remotecall_wait(build_many_factors!, w, grouped_idxs[w], K, M, grouped_xis[w])
    end

    return nothing
end

"""
    solve_block_all_xis(owner, xis, B)

Solve `(K - ξ_i M) X_i = B` for all `ξ_i` using already cached factors.

Arguments
---------
- `owner[i]`: worker that owns the factorization for shift `i`
- `xis`: shift vector, only used for ordering/length
- `B`: fixed RHS block, vector or dense matrix

Returns
-------
A vector `Xs` such that `Xs[i]` is the solution block corresponding to `ξ_i`.
So the ordering of `Xs` matches the ordering of `xis`.
"""
function solve_block_all_xis(owner::Dict{Int,Int}, xis, B)
    ws = unique(values(owner))

    grouped_idxs = Dict(w => Int[] for w in ws)
    for i in eachindex(xis)
        push!(grouped_idxs[owner[i]], i)
    end

    Xs = Vector{Any}(undef, length(xis))

    @sync for w in ws
        idxs_w = grouped_idxs[w]
        isempty(idxs_w) && continue

        @async begin
            Xs_w = remotecall_fetch(solve_same_rhs_with_factors, w, idxs_w, B)
            for (j, i) in enumerate(idxs_w)
                Xs[i] = Xs_w[j]
            end
        end
    end

    return Xs
end

"""
    solve_columns_all_xis(owner, xis, B)
    Solve
        (K - ξ_i M) x_i = b_i

for all shifts `ξ_i`, where `b_i` is column `i` of the block `B`.

Arguments
---------
- `owner[i]`: worker that owns the factorization for shift `i`
- `xis`: shift vector, only used for ordering/length
- `B`: block whose i-th column is the right-hand side for shift `i`

Returns
-------
A matrix `X` whose i-th column is the solution vector corresponding to `ξ_i`.
The column ordering of `X` matches the ordering of `xis`.
"""
function solve_columns_all_xis(owner::Dict{Int,Int}, xis, B)
    length(xis) == size(B, 2) || throw(ArgumentError("B must have as many columns as xis"))

    ws = unique(values(owner))

    grouped_idxs = Dict(w => Int[] for w in ws)
    for i in eachindex(xis)
        push!(grouped_idxs[owner[i]], i)
    end

    T = promote_type(eltype(B), eltype(xis))
    X = Matrix{T}(undef, size(B, 1), length(xis))

    @sync for w in ws
        idxs_w = grouped_idxs[w]
        isempty(idxs_w) && continue

        @async begin
            xs_w = remotecall_fetch(solve_matching_columns_with_factors, w, idxs_w, B)
            for (j, i) in enumerate(idxs_w)
                X[:, i] = xs_w[j]
            end
        end
    end

    return X
end



"""
    free_factors!()

Free cached MUMPS factors on all workers, but keep MPI alive.
"""
function free_factors!()
    for w in workers()
        remotecall_wait(_free_local_factors!, w)
    end
    return nothing
end

"""
    finalize_workers!()

Free factors, finalize MPI on workers, then remove workers.
"""
function finalize_workers!()
    for w in workers()
        remotecall_wait(_shutdown_local_worker!, w)
    end
    rmprocs(workers())
    return nothing
end

end
