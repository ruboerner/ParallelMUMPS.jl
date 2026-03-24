# ParallelMUMPS

[![Build Status](https://github.com/ruboerner/ParallelMUMPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ruboerner/ParallelMUMPS.jl/actions/workflows/CI.yml?query=branch%3Amain)

`ParallelMUMPS.jl` provides a small distributed wrapper around [`MUMPS.jl`](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) for solving families of shifted sparse systems of the form

```math
(K - \xi_i M) X_i = B,
\qquad i = 1,\dots,n_\xi,
```

where

- `K` and `M` are sparse matrices,
- `xis` is a vector of shifts,
- `B` is a fixed right-hand side, possibly with many columns.

The package is designed for the case where

- one factorization is needed per shift,
- the factorizations should be distributed across Julia workers,
- the same factorizations are reused for repeated solves.

## Features

- distributed factorization over shifts using `Distributed`
- one cached MUMPS factorization per shift
- grouped factorization and grouped solves by worker ownership
- support for a fixed dense RHS block `B`
- explicit cleanup of cached MUMPS factors and MPI state

## Problem type

The main use case is a family of systems

```math
A_i = K - \xi_i M,
\qquad
X_i = A_i^{-1} B.
```

The returned solution container is ordered so that `Xs[i]` corresponds to `xis[i]`.

## Installation

From Julia:

```julia
using Pkg
Pkg.add(url="https://github.com/ruboerner/ParallelMUMPS.jl")
```

For development:

```julia
using Pkg
Pkg.develop(url="https://github.com/USERNAME/ParallelMUMPS.jl")
```

## Requirements

- `MPI.jl`
- `MUMPS.jl`
- Julia standard libraries `Distributed`, `SparseArrays`, `LinearAlgebra`

## Exported API

The package currently exports:

- `init_workers!()`
- `factorize_shifts_grouped!(owner, K, M, xis)`
- `solve_block_all_xis(owner, xis, B)`
- `free_factors!()`
- `finalize_workers!()`

## Basic usage

```julia
using Distributed
using SparseArrays
using LinearAlgebra
using ParallelMUMPS

addprocs(2)

@everywhere using ParallelMUMPS

init_workers!()

n = 50
K = sparse(sprand(ComplexF64, n, n, 0.05) + 20.0I)
M = sparse(sprand(ComplexF64, n, n, 0.05) + 2.0I)
xis = ComplexF64[0.1 + 0.1im, 0.2 + 0.1im, 0.3 + 0.1im]

ws = workers()
owner = Dict(i => ws[mod1(i, length(ws))] for i in eachindex(xis))

factorize_shifts_grouped!(owner, K, M, xis)

B = rand(ComplexF64, n, 3)
Xs = solve_block_all_xis(owner, xis, B)

for i in eachindex(xis)
    A = K - xis[i] * M
    println("shift $i residual = ", norm(A * Xs[i] - B) / norm(B))
end

free_factors!()
finalize_workers!()
```

## Ownership model

Each shift index `i` is assigned to a worker through the dictionary

```julia
owner[i] => worker_id
```

The factorization for shift `i` is built and stored on that worker. Solves are then grouped by worker so that remote-call overhead is reduced.

## Notes on sparsity patterns

For best performance, `K` and `M` should have identical sparse patterns. The factorization path reuses a sparse workspace matrix and overwrites its numerical values in place when assembling

```math
K - \xi_i M.
```

The package also supports the more general case where `K` and `M` do **not** share the same CSC structure. In that situation, the shifted matrix is assembled on the combined sparsity pattern internally. This is more flexible, but may lead to higher memory use and more allocations than the matched-pattern case.

So, in summary:

- matching sparsity patterns: preferred for performance
- different sparsity patterns: fully supported, but usually less efficient

## Repeated solves

The intended workflow is:

1.	distribute shifts across workers,
2.	factorize all shifted systems once,
3.	reuse the cached factors for repeated calls to `solve_block_all_xis`.

This is especially useful when `B` has many columns.

## Updating `M`

If `M` changes during, e.g., an outer optimization loop, the current safe strategy is to rebuild the shifted factorizations with the new matrix `Mnew`:

```julia
factorize_shifts_grouped!(owner, K, Mnew, xis)
```

## Testing

From the Julia REPL:

```julia
] test ParallelMUMPS
```

## Caveats

- `MPI` is initialized explicitly on workers via init_workers!().
- `MUMPS` factors are stored in worker-local state.
- `finalize_workers!()` finalizes MPI on workers and removes the workers.
- After `finalize_workers!()`, add workers again before reusing the package in the same Julia session.

