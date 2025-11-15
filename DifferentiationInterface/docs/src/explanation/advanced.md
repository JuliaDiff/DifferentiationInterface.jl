# Advanced features

## Sparsity

When faced with sparse Jacobian or Hessian matrices, one can take advantage of their sparsity pattern to speed up the computation.
DifferentiationInterface does this automatically if you pass a backend of type [`AutoSparse`](@extref ADTypes.AutoSparse).

!!! tip

    To know more about sparse AD, read the survey [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711) (Gebremedhin et al., 2005).

### `AutoSparse` object

`AutoSparse` backends only support [`jacobian`](@ref) and [`hessian`](@ref) (as well as their variants), because other operators do not output matrices.
An `AutoSparse` backend must be constructed from three ingredients:

 1. An underlying (dense) backend, which can be [`SecondOrder`](@ref) or anything from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)

 2. A sparsity pattern detector following the [`ADTypes.AbstractSparsityDetector`](@extref ADTypes.AbstractSparsityDetector) interface, such as:

      + [`TracerSparsityDetector`](@extref SparseConnectivityTracer.TracerSparsityDetector) from [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl)
      + [`SymbolicsSparsityDetector`](@extref Symbolics.SymbolicsSparsityDetector) from [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
      + [`DenseSparsityDetector`](@ref) from DifferentiationInterface.jl (beware that this detector only gives a locally valid pattern)
      + [`KnownJacobianSparsityDetector`](@extref ADTypes.KnownJacobianSparsityDetector) or [`KnownHessianSparsityDetector`](@extref ADTypes.KnownHessianSparsityDetector) from [ADTypes.jl](https://github.com/SciML/ADTypes.jl) (if you already know the pattern)

 3. A coloring algorithm following the [`ADTypes.AbstractColoringAlgorithm`](@extref ADTypes.AbstractColoringAlgorithm) interface, such as those from [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl):

      + [`GreedyColoringAlgorithm`](@extref SparseMatrixColorings.GreedyColoringAlgorithm) (our generic recommendation, don't forget to tune the `order` parameter)
      + [`ConstantColoringAlgorithm`](@extref SparseMatrixColorings.ConstantColoringAlgorithm) (if you have already computed the optimal coloring and always want to return it)
      + [`OptimalColoringAlgorithm`](@extref SparseMatrixColorings.OptimalColoringAlgorithm) (if you have a low-dimensional matrix for which you want to know the best possible coloring)

!!! note

    Symbolic backends have built-in sparsity handling, so `AutoSparse(AutoSymbolics())` and `AutoSparse(AutoFastDifferentiation())` do not need additional configuration for pattern detection or coloring.

### Reusing sparse preparation

The preparation step of `jacobian` or `hessian` with an `AutoSparse` backend can be long, because it needs to detect the sparsity pattern and perform a matrix coloring.
But after preparation, the more zeros are present in the matrix, the greater the speedup will be compared to dense differentiation.

!!! danger

    The result of preparation for an `AutoSparse` backend cannot be reused if the sparsity pattern changes.
    In particular, during preparation, make sure to pick input and context values that do not give rise to exceptional patterns (e.g. with too many zeros because of a multiplication with a constant `c = 0`, which may then be non-zero later on). Random values are usually a better choice during sparse preparation.

### Tuning the coloring algorithm

The complexity of sparse Jacobians or Hessians grows with the number of distinct colors in a coloring of the sparsity pattern.
To reduce this number of colors, [`GreedyColoringAlgorithm`](@extref SparseMatrixColorings.GreedyColoringAlgorithm) has two main settings: the order used for vertices and the decompression method.
Depending on your use case, you may want to modify either of these options to increase performance.
See the documentation of [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl) for details.

### Mixed mode

When a Jacobian matrix has both dense rows and dense columns, it can be more efficient to use "mixed-mode" differentiation, a mixture of forward and reverse.
The associated bidirectional coloring algorithm automatically decides how to cover the Jacobian using a set of columns (computed in forward mode) plus a set of rows (computed in reverse mode).
This behavior is triggered as soon as you put a [`MixedMode`](@ref) object inside `AutoSparse`, like so:

```julia
AutoSparse(
    MixedMode(forward_backend, reverse_backend); sparsity_detector, coloring_algorithm
)
```

At the moment, mixed mode tends to work best (output fewer colors) when the [`GreedyColoringAlgorithm`](@extref SparseMatrixColorings.GreedyColoringAlgorithm) is provided with a [`RandomOrder`](@extref SparseMatrixColorings.RandomOrder) instead of the usual [`NaturalOrder`](@extref SparseMatrixColorings.NaturalOrder), and when "post-processing" is activated after coloring.
For full reproducibility, you should use a random number generator from [StableRNGs.jl](https://github.com/JuliaRandom/StableRNGs.jl).
Thus, the right setup looks like:

```julia
using StableRNGs

seed = 3
coloring_algorithm = GreedyColoringAlgorithm(
    RandomOrder(StableRNG(seed), seed); postprocessing=true
)
```

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
