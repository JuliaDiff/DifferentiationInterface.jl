abstract type FromPrimitive{inplace} <: AbstractADType end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
inplace_support(::FromPrimitive{true}) = InPlaceSupported()
inplace_support(::FromPrimitive{false}) = InPlaceNotSupported()

function pick_batchsize(fromprim::FromPrimitive, N::Integer)
    return pick_batchsize(fromprim.backend, N)
end

"""
    AutoForwardFromPrimitive

Wrapper which forces a given backend to act as a reverse-mode backend.

Used in internal testing.
"""
struct AutoForwardFromPrimitive{inplace,B<:AbstractADType} <: FromPrimitive{inplace}
    backend::B
end

function AutoForwardFromPrimitive(backend::AbstractADType; inplace=true)
    return AutoForwardFromPrimitive{inplace,typeof(backend)}(backend)
end

ADTypes.mode(::AutoForwardFromPrimitive) = ADTypes.ForwardMode()

function threshold_batchsize(
    fromprim::AutoForwardFromPrimitive{inplace}, dimension::Integer
) where {inplace}
    return AutoForwardFromPrimitive(
        threshold_batchsize(fromprim.backend, dimension); inplace
    )
end

struct FromPrimitivePushforwardPrep{E<:PushforwardPrep} <: PushforwardPrep
    pushforward_prep::E
end

function prepare_pushforward(
    f::F, fromprim::AutoForwardFromPrimitive, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    primitive_prep = prepare_pushforward(f, fromprim.backend, x, tx, contexts...)
    return FromPrimitivePushforwardPrep(primitive_prep)
end

function prepare_pushforward(
    f!::F, y, fromprim::AutoForwardFromPrimitive, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    primitive_prep = prepare_pushforward(f!, y, fromprim.backend, x, tx, contexts...)
    return FromPrimitivePushforwardPrep(primitive_prep)
end

function value_and_pushforward(
    f::F,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(
        f, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward(
    f!::F,
    y,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(
        f!, y, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(
        f, ty, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(
        f!, y, ty, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

"""
    AutoReverseFromPrimitive

Wrapper which forces a given backend to act as a reverse-mode backend.

Used in internal testing.
"""
struct AutoReverseFromPrimitive{inplace,B<:AbstractADType} <: FromPrimitive{inplace}
    backend::B
end

function AutoReverseFromPrimitive(backend::AbstractADType; inplace=true)
    return AutoReverseFromPrimitive{inplace,typeof(backend)}(backend)
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

function threshold_batchsize(
    fromprim::AutoReverseFromPrimitive{inplace}, dimension::Integer
) where {inplace}
    return AutoReverseFromPrimitive(
        threshold_batchsize(fromprim.backend, dimension); inplace
    )
end

struct FromPrimitivePullbackPrep{E<:PullbackPrep} <: PullbackPrep
    pullback_prep::E
end

function prepare_pullback(
    f::F, fromprim::AutoReverseFromPrimitive, x, ty::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    primitive_prep = prepare_pullback(f, fromprim.backend, x, ty, contexts...)
    return FromPrimitivePullbackPrep(primitive_prep)
end

function prepare_pullback(
    f!::F, y, fromprim::AutoReverseFromPrimitive, x, ty::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    primitive_prep = prepare_pullback(f!, y, fromprim.backend, x, ty, contexts...)
    return FromPrimitivePullbackPrep(primitive_prep)
end

function value_and_pullback(
    f::F,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(f, prep.pullback_prep, fromprim.backend, x, ty, contexts...)
end

function value_and_pullback(
    f!::F,
    y,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(
        f!, y, prep.pullback_prep, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f::F,
    tx::NTuple,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(
        f, tx, prep.pullback_prep, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f!::F,
    y,
    tx::NTuple,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(
        f!, y, tx, prep.pullback_prep, fromprim.backend, x, ty, contexts...
    )
end
