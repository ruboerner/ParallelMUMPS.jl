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

mutable struct WorkerState
    factors_f64::Dict{Int,Mumps{Float64}}
    factors_c64::Dict{Int,Mumps{ComplexF64}}
end

const STATE = WorkerState(
    Dict{Int,Mumps{Float64}}(),
    Dict{Int,Mumps{ComplexF64}}(),
)

factor_dict(::Type{Float64}) = STATE.factors_f64
factor_dict(::Type{ComplexF64}) = STATE.factors_c64
factor_dict(::Type{T}) where {T} =
    throw(ArgumentError("Unsupported MUMPS scalar type: $T"))

function _delete_factor!(i::Integer)
    ii = Int(i)
    if haskey(STATE.factors_f64, ii)
        finalize(STATE.factors_f64[ii])
        delete!(STATE.factors_f64, ii)
    end
    if haskey(STATE.factors_c64, ii)
        finalize(STATE.factors_c64[ii])
        delete!(STATE.factors_c64, ii)
    end
    return nothing
end

function _factor_dict_for_idxs(idxs::AbstractVector{<:Integer})
    isempty(idxs) && throw(ArgumentError("idxs must be non-empty"))
    i0 = Int(first(idxs))
    if haskey(STATE.factors_f64, i0)
        return STATE.factors_f64
    elseif haskey(STATE.factors_c64, i0)
        return STATE.factors_c64
    else
        throw(KeyError(i0))
    end
end

# -----------------------------------------------------------------------------
# Worker lifecycle
# -----------------------------------------------------------------------------

function _init_local_worker!()
    if !MPI.Initialized()
        MPI.Init()
    end
    return nothing
end

function _free_local_factors!()
    for m in values(STATE.factors_f64)
        finalize(m)
    end
    for m in values(STATE.factors_c64)
        finalize(m)
    end
    empty!(STATE.factors_f64)
    empty!(STATE.factors_c64)
    return nothing
end

function _shutdown_local_worker!()
    _free_local_factors!()
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Assembly helpers
# -----------------------------------------------------------------------------

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

same_pattern(K::SparseMatrixCSC, M::SparseMatrixCSC) =
    K.colptr == M.colptr && K.rowval == M.rowval

# -----------------------------------------------------------------------------
# Factorization
# -----------------------------------------------------------------------------

function build_factor!(i::Integer, A::SparseMatrixCSC{T,Ti}) where {T<:Union{Float64,ComplexF64},Ti}
    _delete_factor!(i)
    icntl = get_icntl(verbose=false)
    m = Mumps{T}(mumps_unsymmetric, icntl, default_cntl64)
    associate_matrix!(m, A)
    factorize!(m)
    factor_dict(T)[Int(i)] = m
    return nothing
end

function build_factor!(i::Integer,
                       K::SparseMatrixCSC{TK,Ti},
                       M::SparseMatrixCSC{TM,Ti},
                       ξ) where {TK<:Union{Float64,ComplexF64},TM<:Union{Float64,ComplexF64},Ti}

    T = promote_type(TK, TM, typeof(ξ))
    T <: Union{Float64,ComplexF64} ||
        throw(ArgumentError("Unsupported factor scalar type: $T"))

    A = K - ξ * M
    build_factor!(i, A)
end

function build_many_factors!(idxs::AbstractVector{<:Integer},
                             K::SparseMatrixCSC{TK,Ti},
                             M::SparseMatrixCSC{TM,Ti},
                             xis::AbstractVector) where {TK<:Union{Float64,ComplexF64},TM<:Union{Float64,ComplexF64},Ti}

    length(idxs) == length(xis) || throw(ArgumentError("idxs and xis must have same length"))

    Twork = promote_type(TK, TM, eltype(xis))
    Twork <: Union{Float64,ComplexF64} ||
        throw(ArgumentError("Unsupported factor scalar type: $Twork"))

    if same_pattern(K, M)
        Awork = SparseMatrixCSC(
            size(K,1), size(K,2),
            copy(K.colptr),
            copy(K.rowval),
            Vector{Twork}(undef, nnz(K))
        )

        for k in eachindex(idxs)
            _assemble_shift!(Awork, K, M, xis[k])
            build_factor!(idxs[k], Awork)
        end
    else
        for k in eachindex(idxs)
            build_factor!(idxs[k], K, M, xis[k])
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Local solves (typed, allocation-aware)
# -----------------------------------------------------------------------------

function _solve_same_rhs_with_factors(idxs::AbstractVector{<:Integer},
                                      b::AbstractVector,
                                      cache::Dict{Int,Mumps{S}}) where {S}
    Xs = Vector{Vector{S}}(undef, length(idxs))
    rhs = Vector{S}(undef, length(b))

    for k in eachindex(idxs)
        copyto!(rhs, b)
        m = cache[Int(idxs[k])]
        associate_rhs!(m, rhs)
        solve!(m)
        Xs[k] = vec(copy(get_solution(m)))
        MUMPS.set_job!(m, 4)
    end
    return Xs
end

function _solve_same_rhs_with_factors(idxs::AbstractVector{<:Integer},
                                      B::AbstractMatrix,
                                      cache::Dict{Int,Mumps{S}}) where {S}
    Xs = Vector{Matrix{S}}(undef, length(idxs))
    rhs = Matrix{S}(undef, size(B, 1), size(B, 2))

    for k in eachindex(idxs)
        copyto!(rhs, B)
        m = cache[Int(idxs[k])]
        associate_rhs!(m, rhs)
        solve!(m)
        Xs[k] = copy(get_solution(m))
        MUMPS.set_job!(m, 4)
    end
    return Xs
end

function solve_same_rhs_with_factors(idxs::AbstractVector{<:Integer}, B::AbstractVecOrMat)
    cache = _factor_dict_for_idxs(idxs)
    return _solve_same_rhs_with_factors(idxs, B, cache)
end

function _solve_matching_columns_with_factors(idxs::AbstractVector{<:Integer},
                                              B::AbstractMatrix,
                                              cache::Dict{Int,Mumps{S}}) where {S}
    size(B, 2) >= maximum(idxs) || throw(ArgumentError("B must have at least one column per shift index"))

    X = Matrix{S}(undef, size(B, 1), length(idxs))
    bi = Vector{S}(undef, size(B, 1))

    for k in eachindex(idxs)
        i = Int(idxs[k])
        m = cache[i]
        copyto!(bi, view(B, :, i))
        associate_rhs!(m, bi)
        solve!(m)
        X[:, k] = vec(copy(get_solution(m)))
        MUMPS.set_job!(m, 4)
    end
    return X
end

function solve_matching_columns_with_factors(idxs::AbstractVector{<:Integer}, B::AbstractMatrix)
    cache = _factor_dict_for_idxs(idxs)
    return _solve_matching_columns_with_factors(idxs, B, cache)
end

# -----------------------------------------------------------------------------
# Distributed API
# -----------------------------------------------------------------------------

function init_workers!()
    ws = workers()
    isempty(ws) && error("No workers. Call addprocs(...) first.")

    @sync for w in ws
        @async remotecall_wait(_init_local_worker!, w)
    end

    return nothing
end

function factorize_shifts_grouped!(owner::Dict{Int,Int}, K, M, xis)
    ws = unique(values(owner))

    grouped_idxs = Dict(w => Int[] for w in ws)
    grouped_xis = Dict(w => Vector{eltype(xis)}() for w in ws)

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
            for (j,i) in enumerate(idxs_w)
                Xs[i] = Xs_w[j]
            end
        end
    end

    return Xs
end

function solve_columns_all_xis(owner::Dict{Int,Int}, xis, B)
    length(xis) == size(B,2) || throw(ArgumentError("B mismatch"))

    ws = unique(values(owner))
    grouped_idxs = Dict(w => Int[] for w in ws)

    for i in eachindex(xis)
        push!(grouped_idxs[owner[i]], i)
    end

    T = promote_type(eltype(B), eltype(xis))
    X = Matrix{T}(undef, size(B,1), length(xis))

    @sync for w in ws
        idxs_w = grouped_idxs[w]
        isempty(idxs_w) && continue

        @async begin
            X_w = remotecall_fetch(solve_matching_columns_with_factors, w, idxs_w, B)
            for (j,i) in enumerate(idxs_w)
                X[:,i] = view(X_w,:,j)
            end
        end
    end

    return X
end

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

function free_factors!()
    for w in workers()
        remotecall_wait(_free_local_factors!, w)
    end
end

function finalize_workers!()
    for w in workers()
        remotecall_wait(_shutdown_local_worker!, w)
    end
    rmprocs(workers())
end

end