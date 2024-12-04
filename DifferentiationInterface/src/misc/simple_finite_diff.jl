"""
    AutoSimpleFiniteDiff <: ADTypes.AbstractADType

Forward mode backend based on the finite difference `(f(x + ε) - f(x)) / ε`, with artificial chunk size to mimick ForwardDiff.

# Constructor

    AutoSimpleFiniteDiff(ε=1e-5; chunksize=nothing)
"""
struct AutoSimpleFiniteDiff{chunksize,T<:Real} <: AbstractADType
    ε::T
end

function AutoSimpleFiniteDiff(ε=1e-5; chunksize=nothing)
    return AutoSimpleFiniteDiff{chunksize,typeof(ε)}(ε)
end

ADTypes.mode(::AutoSimpleFiniteDiff) = ForwardMode()
check_available(::AutoSimpleFiniteDiff) = true
inplace_support(::AutoSimpleFiniteDiff) = InPlaceSupported()

function BatchSizeSettings(::AutoSimpleFiniteDiff{nothing}, N::Integer)
    B = min(12, N)
    singlebatch = B == N
    aligned = N % B == 0
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

function BatchSizeSettings(::AutoSimpleFiniteDiff{chunksize}, N::Integer) where {chunksize}
    @assert chunksize <= N
    B = chunksize
    singlebatch = B == N
    aligned = N % B == 0
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

function threshold_batchsize(
    backend::AutoSimpleFiniteDiff{chunksize1}, chunksize2::Integer
) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoSimpleFiniteDiff(backend.ε; chunksize)
end

function prepare_pushforward(
    f::F, ::AutoSimpleFiniteDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return NoPushforwardPrep()
end

function prepare_pushforward(
    f!::F, y, ::AutoSimpleFiniteDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return NoPushforwardPrep()
end

function value_and_pushforward(
    f::F,
    ::NoPushforwardPrep,
    backend::AutoSimpleFiniteDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    ε = eltype(x)(backend.ε)
    y = f(x, map(unwrap, contexts)...)
    ty = map(tx) do dx
        yε = f(x + ε * dx, map(unwrap, contexts)...)
        y0 = f(x, map(unwrap, contexts)...)
        (yε - y0) / ε
    end
    return y, ty
end

function value_and_pushforward(
    f!::F,
    y,
    ::NoPushforwardPrep,
    backend::AutoSimpleFiniteDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    ε = eltype(x)(backend.ε)
    ty = map(tx) do dx
        f!(y, x + ε * dx, map(unwrap, contexts)...)
        yε = copy(y)
        f!(y, x, map(unwrap, contexts)...)
        y0 = copy(y)
        (yε - y0) / ε
    end
    f!(y, x, map(unwrap, contexts)...)
    return y, ty
end
