struct ReturnZero{T}
    template::T
end

(rz::ReturnZero)(i) = zero(rz.template)

_zero!(x::AbstractArray{T}) where {T} = fill!(x, zero(T))

## Forward

"""
    AutoZeroForward <: ADTypes.AbstractADType

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: AbstractADType end

ADTypes.mode(::AutoZeroForward) = ForwardMode()
check_available(::AutoZeroForward) = true
inplace_support(::AutoZeroForward) = InPlaceSupported()

function prepare_pushforward_nokwarg(
        strict::Val, f::F, backend::AutoZeroForward, x, tx::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    return NoPushforwardPrep(_sig)
end

function prepare_pushforward_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoZeroForward,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f!, y, backend, x, tx, contexts...; strict)
    return NoPushforwardPrep(_sig)
end

function value_and_pushforward(
        f::F,
        prep::NoPushforwardPrep,
        backend::AutoZeroForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    y = f(x, map(unwrap, contexts)...)
    ty = map(ReturnZero(y), tx)
    return y, ty
end

function value_and_pushforward(
        f!::F,
        y,
        prep::NoPushforwardPrep,
        backend::AutoZeroForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    ty = map(ReturnZero(y), tx)
    return y, ty
end

function value_and_pushforward!(
        f::F,
        ty::NTuple,
        prep::NoPushforwardPrep,
        backend::AutoZeroForward,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    y = f(x, map(unwrap, contexts)...)
    for b in eachindex(ty)
        _zero!(ty[b])
    end
    return y, ty
end

function value_and_pushforward!(
        f!::F,
        y,
        ty::NTuple,
        prep::NoPushforwardPrep,
        backend::AutoZeroForward,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    for b in eachindex(ty)
        _zero!(ty[b])
    end
    return y, ty
end

## Reverse

"""
    AutoZeroReverse <: ADTypes.AbstractADType

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: AbstractADType end

ADTypes.mode(::AutoZeroReverse) = ReverseMode()
check_available(::AutoZeroReverse) = true
inplace_support(::AutoZeroReverse) = InPlaceSupported()

function prepare_pullback_nokwarg(
        strict::Val, f::F, backend::AutoZeroReverse, x, ty::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, ty, contexts...; strict)
    return NoPullbackPrep(_sig)
end

function prepare_pullback_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoZeroReverse,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f!, y, backend, x, ty, contexts...; strict)
    return NoPullbackPrep(_sig)
end

function value_and_pullback(
        f::F,
        prep::NoPullbackPrep,
        backend::AutoZeroReverse,
        x,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f, prep, backend, x, ty, contexts...)
    y = f(x, map(unwrap, contexts)...)
    tx = ntuple(ReturnZero(x), Val(B))
    return y, tx
end

function value_and_pullback(
        f!::F,
        y,
        prep::NoPullbackPrep,
        backend::AutoZeroReverse,
        x,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    tx = ntuple(ReturnZero(x), Val(B))
    return y, tx
end

function value_and_pullback!(
        f::F,
        tx::NTuple,
        prep::NoPullbackPrep,
        backend::AutoZeroReverse,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, ty, contexts...)
    y = f(x, map(unwrap, contexts)...)
    for b in eachindex(tx)
        _zero!(tx[b])
    end
    return y, tx
end

function value_and_pullback!(
        f!::F,
        y,
        tx::NTuple,
        prep::NoPullbackPrep,
        backend::AutoZeroReverse,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    for b in eachindex(tx)
        _zero!(tx[b])
    end
    return y, tx
end
