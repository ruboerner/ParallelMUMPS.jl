using Test
using Distributed
using ParallelMUMPS
using SparseArrays
using LinearAlgebra

@testset "ParallelMUMPS" begin
    addprocs(2)
	@everywhere using ParallelMUMPS

    ParallelMUMPS.init_workers!()

    n = 50
    K = sparse(sprand(ComplexF64, n, n, 0.05) + 20.0I)
    M = K 
    xis = ComplexF64[0.1 + 0.1im, 0.2 + 0.1im, 0.3 + 0.1im]

    ws = workers()
    owner = Dict(i => ws[mod1(i, length(ws))] for i in eachindex(xis))

    ParallelMUMPS.factorize_shifts_grouped!(owner, K, M, xis)

    B = rand(ComplexF64, n, 2)
    Xs = ParallelMUMPS.solve_block_all_xis(owner, xis, B)

    for i in eachindex(xis)
        A = K - xis[i] * M
        @test norm(A * Xs[i] - B) / norm(B) < 1e-8
    end

    M = sparse(sprand(ComplexF64, n, n, 0.05) + 2.0I)
    ParallelMUMPS.factorize_shifts_grouped!(owner, K, M, xis)

    Xs = ParallelMUMPS.solve_block_all_xis(owner, xis, B)

    for i in eachindex(xis)
        A = K - xis[i] * M
        @test norm(A * Xs[i] - B) / norm(B) < 1e-8
    end

    B = rand(ComplexF64, n, length(xis))
    Xs = ParallelMUMPS.solve_columns_all_xis(owner, xis, B)

    @info size(Xs)
    @info size(B)

    for i in eachindex(xis)
        A = K - xis[i] * M
        @test norm(A * Xs[:, i] - B[:, i]) / norm(B[:, i]) < 1e-8
    end

    ParallelMUMPS.free_factors!()
    ParallelMUMPS.finalize_workers!()
end
