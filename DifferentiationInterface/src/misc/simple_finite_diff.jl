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
check_operator_overloading(::AutoSimpleFiniteDiff) = false

function pick_batchsize(::AutoSimpleFiniteDiff{nothing}, N::Integer)
    B = reasonable_batchsize(N, 12)
    return BatchSizeSettings{B}(N)
end

function pick_batchsize(::AutoSimpleFiniteDiff{chunksize}, N::Integer) where {chunksize}
    return BatchSizeSettings{chunksize}(N)
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
        y1 = f(x + ε * dx, map(unwrap, contexts)...)
        y0 = f(x - ε * dx, map(unwrap, contexts)...)
        (y1 - y0) / 2ε
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
        y1 = copy(y)
        f!(y, x - ε * dx, map(unwrap, contexts)...)
        y0 = copy(y)
        (y1 - y0) / 2ε
    end
    f!(y, x, map(unwrap, contexts)...)
    return y, ty
end
