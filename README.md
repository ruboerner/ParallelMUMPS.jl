# ParallelMUMPS

[![Build Status](https://github.com/ruboerner/ParallelMUMPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ruboerner/ParallelMUMPS.jl/actions/workflows/CI.yml?query=branch%3Amain)

`ParallelMUMPS.jl` provides a small distributed wrapper around [`MUMPS.jl`](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) for solving families of shifted sparse systems of the form

```math
(\mathbf K - \xi_i \mathbf M) \mathbf X_i = \mathbf B,
\qquad i = 1,\dots,n_\xi,
```

where

- `K` and `M` are sparse real-valued matrices,
- `xis` is a vector of (complex) shifts,
- `B` is either a fixed right-hand vector, or a block with as many columns as entries in `xis`,
- `X` is a block of (complex) solutions, columns ordered so that `X[:, i]` corresponds to `xis[i]`

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
\mathbf A_i = \mathbf K - \xi_i \mathbf M,
\qquad
\mathbf X_i = \mathbf A_i^{-1} \mathbf B.
```

The returned solution container is ordered so that `X[i]` corresponds to `xis[i]`.

## Installation

From Julia:

```julia
using Pkg
Pkg.add(url="https://github.com/ruboerner/ParallelMUMPS.jl")
```

For development:

```julia
using Pkg
Pkg.develop(url="https://github.com/ruboerner/ParallelMUMPS.jl")
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
- `solve_columns_all_xis(owner, xis, B)`
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
X = solve_block_all_xis(owner, xis, B)

for i in eachindex(xis)
    A = K - xis[i] * M
    println("shift $i residual = ", norm(A * X[i] - B) / norm(B))
end

free_factors!()
finalize_workers!()
```

## Ownership model

Each shift index `i` is assigned to a worker through the dictionary

```julia
owner[i] => worker_id
```

The factorization for shift `i` is built and stored on that worker.

If the number of workers is smaller than the number of shifts, the shifts are still processed in parallel, but in grouped batches. Each worker handles its assigned batch sequentially, while the batches themselves run concurrently across workers. The grouping has to be defined once as follows:

```julia
ws = workers()
owner = Dict(i => ws[mod1(i, length(ws))] for i in eachindex(xis))
```


## Notes on sparsity patterns

For best performance, `K` and `M` should have identical sparse patterns. The factorization path reuses a sparse workspace matrix and overwrites its numerical values in place when assembling

```math
\mathbf A_i := \mathbf K - \xi_i \mathbf M.
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

### Tuning workers and threads

For a machine with many cores, performance depends on the balance between the number of Julia workers (parallelism over shifts) and the number of OpenMP threads used inside each MUMPS factorization.

A practical rule is to choose the number of workers and OpenMP threads so that
`n_workers * n_threads` is somewhat below the total number of physical cores, leaving headroom for system processes and runtime overhead.

So, for 16 shifts on a 96-core machine, it is sensible to benchmark configurations such as 16×5, 10×8, or 8×10 (workers × OpenMP threads). Using more workers than shifts is not beneficial.

In general, the number of workers should not exceed the number of shifts, since there is at most one independent factorization per shift. If fewer workers than shifts are used, the shifts are processed in grouped batches: each worker handles its assigned batch sequentially, while batches run concurrently across workers.

A practical rule is to choose the number of workers and OpenMP threads so that

```math
n_{\mathrm{workers}} \times n_{\mathrm{threads}} \approx n_{\mathrm{cores}}.
```

Benchmarking a few such combinations is recommended, since the optimal balance depends on the size of the sparse direct factorization and the available memory bandwidth.

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
