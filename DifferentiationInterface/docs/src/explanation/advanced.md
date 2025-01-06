# Advanced features

## Contexts

### Additional arguments

For all operators provided DifferentiationInterface, there can be only one differentiated (or "active") argument, which we call `x`.
However, the release v0.6 introduced the possibility of additional "context" arguments, which are not differentiated but still passed to the function after `x`.

Contexts can be useful if you have a function `y = f(x, a, b, c, ...)` or `f!(y, x, a, b, c, ...)` and you want derivatives of `y` with respect to `x` only.
Another option would be creating a closure, but that is sometimes undesirable.

### Types of contexts

Every context argument must be wrapped in a subtype of [`Context`](@ref) and come after the differentiated input `x`.
Right now, there are two kinds of context: [`Constant`](@ref) and [`Cache`](@ref).

!!! warning
    This feature is still experimental and will not be supported by all backends.
    At the moment:
    - `Constant` is supported by all backends except symbolic ones
    - `Cache` is only supported by finite difference backends and [`AutoForwardDiff`](@ref), but it is not yet optimized

Semantically, both of these calls compute the partial gradient of `f(x, c)` with respect to `x`, but they consider `c` differently:

```julia
gradient(f, backend, x, Constant(c))
gradient(f, backend, x, Cache(c))
```

In the first call, `c` is kept unchanged throughout the function evaluation.
In the second call, `c` can be mutated with values computed during the function.

Importantly, one can prepare an operator with an arbitrary value `c'` of the `Constant` (subject to the usual restrictions on preparation).
The values in a provided `Cache` never matter anyway.

## Sparsity

When faced with sparse Jacobian or Hessian matrices, one can take advantage of their sparsity pattern to speed up the computation.
DifferentiationInterface does this automatically if you pass a backend of type [`AutoSparse`](@extref ADTypes.AutoSparse).

!!! tip
    To know more about sparse AD, read the survey [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711) (Gebremedhin et al., 2005).

### `AutoSparse` object

`AutoSparse` backends only support [`jacobian`](@ref) and [`hessian`](@ref) (as well as their variants), because other operators do not output matrices.
An `AutoSparse` backend must be constructed from three ingredients:

1. An underlying (dense) backend, which can be [`SecondOrder`](@ref) or anything from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)
2. A sparsity pattern detector like:
   - [`TracerSparsityDetector`](@extref SparseConnectivityTracer.TracerSparsityDetector) from [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl)
   - [`SymbolicsSparsityDetector`](@extref Symbolics.SymbolicsSparsityDetector) from [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
   - [`DenseSparsityDetector`](@ref) from DifferentiationInterface.jl (beware that this detector only gives a locally valid pattern)
   - [`KnownJacobianSparsityDetector`](@extref ADTypes.KnownJacobianSparsityDetector) or [`KnownHessianSparsityDetector`](@extref ADTypes.KnownHessianSparsityDetector) from [ADTypes.jl](https://github.com/SciML/ADTypes.jl) (if you already know the pattern)
3. A coloring algorithm from [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl), such as:
   - [`GreedyColoringAlgorithm`](@extref SparseMatrixColorings.GreedyColoringAlgorithm) (our generic recommendation)
   - [`ConstantColoringAlgorithm`](@extref SparseMatrixColorings.ConstantColoringAlgorithm) (if you have already computed the optimal coloring and always want to return it)

!!! note
    Symbolic backends have built-in sparsity handling, so `AutoSparse(AutoSymbolics())` and `AutoSparse(AutoFastDifferentiation())` do not need additional configuration for pattern detection or coloring.

### Cost of sparse preparation

The preparation step of `jacobian` or `hessian` with an `AutoSparse` backend can be long, because it needs to detect the sparsity pattern and perform a matrix coloring.
But after preparation, the more zeros are present in the matrix, the greater the speedup will be compared to dense differentiation.

!!! danger
    The result of preparation for an `AutoSparse` backend cannot be reused if the sparsity pattern changes.

### Tuning the coloring algorithm

The complexity of sparse Jacobians or Hessians grows with the number of distinct colors in a coloring of the sparsity pattern.
To reduce this number of colors, [`GreedyColoringAlgorithm`](@ref) has two main settings: the order used for vertices and the decompression method.
Depending on your use case, you may want to modify either of these options to increase performance.
See the documentation of [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl) for details.

## Batch mode

### Multiple tangents

The [`jacobian`](@ref) and [`hessian`](@ref) operators compute matrices by repeatedly applying lower-level operators ([`pushforward`](@ref), [`pullback`](@ref) or [`hvp`](@ref)) to a set of tangents.
The tangents usually correspond to basis elements of the appropriate vector space.
We could call the lower-level operator on each tangent separately, but some packages ([ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)) have optimized implementations to handle multiple tangents at once.

This behavior is often called "vector mode" AD, but we call it "batch mode" to avoid confusion with Julia's `Vector` type.
As a matter of fact, the optimal batch size $B$ (number of simultaneous tangents) is usually very small, so tangents are passed within an `NTuple` and not a `Vector`.
When the underlying vector space has dimension $N$, the operators `jacobian` and `hessian` process $\lceil N / B \rceil$ batches of size $B$ each.

### Optimal batch size

For every backend which does not support batch mode, the batch size is set to $B = 1$.
But for [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff) and [`AutoEnzyme`](@extref ADTypes.AutoEnzyme), more complicated rules apply.
If the backend object has a pre-determined batch size $B_0$, then we always set $B = B_0$.
In particular, this will throw errors when $N < B_0$.
On the other hand, without a pre-determined batch size, we apply backend-specific heuristics to pick $B$ based on $N$.
