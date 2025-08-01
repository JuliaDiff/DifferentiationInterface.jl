"""
    BatchSizeSettings{B,singlebatch,aligned}

Configuration for the batch size deduced from a backend and a sample array of length `N`.

# Type parameters

- `B::Int`: batch size
- `singlebatch::Bool`: whether `B == N` (`B > N` is not allowed)
- `aligned::Bool`: whether `N % B == 0`

# Fields

- `N::Int`: array length
- `A::Int`: number of batches `A = div(N, B, RoundUp)`
- `B_last::Int`: size of the last batch (if `aligned` is `false`)
"""
struct BatchSizeSettings{B,singlebatch,aligned}
    N::Int
    A::Int
    B_last::Int
end

function BatchSizeSettings{B,singlebatch,aligned}(N::Integer) where {B,singlebatch,aligned}
    B > N && throw(ArgumentError("Batch size $B larger than input size $N"))
    A = div(N, B, RoundUp)
    B_last = N % B
    return BatchSizeSettings{B,singlebatch,aligned}(N, A, B_last)
end

function BatchSizeSettings{B}(::Val{N}) where {B,N}
    singlebatch = B == N
    aligned = N % B == 0
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

function BatchSizeSettings{B}(N::Integer) where {B}
    # type-unstable
    singlebatch = B == N
    aligned = N % B == 0
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

"""
    pick_batchsize(backend, N::Integer)

Return a [`BatchSizeSettings`](@ref) appropriate for arrays of length `N` with a given `backend`.
"""
function pick_batchsize(backend::AbstractADType, N::Integer)
    check_batchsize_pickable(backend)
    B = 1
    singlebatch = false
    aligned = true
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

"""
    pick_batchsize(backend, x_or_y::AbstractArray)

Return a [`BatchSizeSettings`](@ref) appropriate for arrays of the same length as `x_or_y` with a given `backend`.

Note that the array in question can be either the input or the output of the function, depending on whether the backend performs forward- or reverse-mode AD.
"""
function pick_batchsize(backend::AbstractADType, x::AbstractArray)
    check_batchsize_pickable(backend)
    return pick_batchsize(backend, length(x))
end

function check_batchsize_pickable(backend::AbstractADType)
    if backend isa SecondOrder
        throw(
            ArgumentError(
                "You should select the batch size for the inner or outer backend of $backend",
            ),
        )
    elseif backend isa AutoSparse
        throw(
            ArgumentError(
                "You should select the batch size for the dense backend of $backend"
            ),
        )
    elseif backend isa MixedMode
        throw(
            ArgumentError(
                "You should select the batch size for the forward or reverse backend of $backend",
            ),
        )
    end
end

"""
    threshold_batchsize(backend::AbstractADType, B::Integer)

If the backend object has a fixed batch size `B0`, return a new backend where the fixed batch size is `min(B0, B)`.
Otherwise, act as the identity.
"""
function threshold_batchsize end

threshold_batchsize(backend::AbstractADType, ::Integer) = backend

function threshold_batchsize(backend::AutoSparse, B::Integer)
    return AutoSparse(
        threshold_batchsize(dense_ad(backend), B);
        sparsity_detector=backend.sparsity_detector,
        coloring_algorithm=backend.coloring_algorithm,
    )
end

function threshold_batchsize(backend::SecondOrder, B::Integer)
    return SecondOrder(
        threshold_batchsize(outer(backend), B), threshold_batchsize(inner(backend), B)
    )
end

"""
    reasonable_batchsize(N::Integer, Bmax::Integer)

Reproduces the heuristic from ForwardDiff to minimize

1. the number of batches necessary to cover an array of length `N`
2. the number of leftover indices in the last partial batch

Source: https://github.com/JuliaDiff/ForwardDiff.jl/blob/ec74fbc32b10bbf60b3c527d8961666310733728/src/prelude.jl#L19-L29
"""
function reasonable_batchsize(N::Integer, Bmax::Integer)
    if N <= Bmax
        return N
    else
        A = div(N, Bmax, RoundUp)
        B = div(N, A, RoundUp)
        return B
    end
end
