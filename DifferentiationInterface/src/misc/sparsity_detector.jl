"""
    DenseSparsityDetector

Sparsity pattern detector satisfying the [detection API](https://sciml.github.io/ADTypes.jl/stable/#Sparse-AD) of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

The nonzeros in a Jacobian or Hessian are detected by computing the relevant matrix with _dense_ AD, and thresholding the entries with a given tolerance (which can be numerically inaccurate).
This process can be very slow, and should only be used if its output can be exploited multiple times to compute many sparse matrices.

!!! danger

    In general, the sparsity pattern you obtain can depend on the provided input `x`. If you want to reuse the pattern, make sure that it is input-agnostic.

!!! warning

    `DenseSparsityDetector` functionality is now located in a package extension, please load the SparseArrays.jl standard library before you use it.

# Fields

  - `backend::AbstractADType` is the dense AD backend used under the hood
  - `atol::Float64` is the minimum magnitude of a matrix entry to be considered nonzero

# Constructor

    DenseSparsityDetector(backend; atol, method=:iterative)

The keyword argument `method::Symbol` can be either:

  - `:iterative`: compute the matrix in a sequence of matrix-vector products (memory-efficient)
  - `:direct`: compute the matrix all at once (memory-hungry but sometimes faster).

Note that the constructor is type-unstable because `method` ends up being a type parameter of the `DenseSparsityDetector` object (this is not part of the API and might change).

# Examples

```jldoctest detector
using ADTypes, DifferentiationInterface, SparseArrays
using ForwardDiff: ForwardDiff

detector = DenseSparsityDetector(AutoForwardDiff(); atol=1e-5, method=:direct)

ADTypes.jacobian_sparsity(diff, rand(5), detector)

# output

4×5 SparseMatrixCSC{Bool, Int64} with 8 stored entries:
 1  1  ⋅  ⋅  ⋅
 ⋅  1  1  ⋅  ⋅
 ⋅  ⋅  1  1  ⋅
 ⋅  ⋅  ⋅  1  1
```

Sometimes the sparsity pattern is input-dependent:

```jldoctest detector
ADTypes.jacobian_sparsity(x -> [prod(x)], rand(2), detector)

# output

1×2 SparseMatrixCSC{Bool, Int64} with 2 stored entries:
 1  1
```

```jldoctest detector
ADTypes.jacobian_sparsity(x -> [prod(x)], [0, 1], detector)

# output

1×2 SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 1  ⋅
```
"""
struct DenseSparsityDetector{method, B} <: ADTypes.AbstractSparsityDetector
    backend::B
    atol::Float64
end

function Base.show(io::IO, detector::DenseSparsityDetector{method}) where {method}
    (; backend, atol) = detector
    return print(
        io,
        DenseSparsityDetector,
        "(",
        repr(backend; context = io),
        "; atol=$atol, method=",
        repr(method; context = io),
        ")",
    )
end

function DenseSparsityDetector(
        backend::AbstractADType; atol::Float64, method::Symbol = :iterative
    )
    if !(method in (:iterative, :direct))
        throw(
            ArgumentError("The keyword `method` must be either `:iterative` or `:direct`.")
        )
    end
    return DenseSparsityDetector{method, typeof(backend)}(backend, atol)
end
