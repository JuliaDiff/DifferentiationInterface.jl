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

function BatchSizeSettings(::AbstractADType, N::Integer)
    B = 1
    singlebatch = false
    aligned = true
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

function BatchSizeSettings(backend::AbstractADType, x::AbstractArray)
    N = length(x)
    return BatchSizeSettings(backend, N)
end

function pick_batchsize(backend::AbstractADType, x_or_N::Union{AbstractArray,Integer})
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
    else
        return BatchSizeSettings(backend, x_or_N)
    end
end

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

function reasonable_batchsize(N::Integer, Bmax::Integer)
    # borrowed from https://github.com/JuliaDiff/ForwardDiff.jl/blob/ec74fbc32b10bbf60b3c527d8961666310733728/src/prelude.jl#L19-L29
    if N <= Bmax
        return N
    else
        A = div(N, Bmax, RoundUp)
        B = div(N, A, RoundUp)
        return B
    end
end
