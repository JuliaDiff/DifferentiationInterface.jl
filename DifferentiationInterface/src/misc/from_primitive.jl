abstract type FromPrimitive{inplace} <: AbstractADType end

check_available(backend::FromPrimitive) = check_available(backend.backend)
inplace_support(::FromPrimitive{true}) = InPlaceSupported()
inplace_support(::FromPrimitive{false}) = InPlaceNotSupported()
function inner_preparation_behavior(backend::FromPrimitive)
    return inner_preparation_behavior(backend.backend)
end

function pick_batchsize(backend::FromPrimitive, N::Integer)
    return pick_batchsize(backend.backend, N)
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
    backend::AutoForwardFromPrimitive{inplace}, dimension::Integer
) where {inplace}
    return AutoForwardFromPrimitive(
        threshold_batchsize(backend.backend, dimension); inplace
    )
end

struct FromPrimitivePushforwardPrep{SIG,E<:PushforwardPrep} <: PushforwardPrep{SIG}
    _sig::Val{SIG}
    pushforward_prep::E
end

function prepare_pushforward(
    strict::Val,
    f::F,
    backend::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
) where {F,C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    primitive_prep = prepare_pushforward(strict, f, backend.backend, x, tx, contexts...)
    return FromPrimitivePushforwardPrep(_sig, primitive_prep)
end

function prepare_pushforward(
    strict::Val,
    f!::F,
    y,
    backend::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
) where {F,C}
    _sig = signature(f!, y, backend, x, tx, contexts...; strict)
    primitive_prep = prepare_pushforward(strict, f!, y, backend.backend, x, tx, contexts...)
    return FromPrimitivePushforwardPrep(_sig, primitive_prep)
end

function value_and_pushforward(
    f::F,
    prep::FromPrimitivePushforwardPrep,
    backend::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    return value_and_pushforward(
        f, prep.pushforward_prep, backend.backend, x, tx, contexts...
    )
end

function value_and_pushforward(
    f!::F,
    y,
    prep::FromPrimitivePushforwardPrep,
    backend::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    return value_and_pushforward(
        f!, y, prep.pushforward_prep, backend.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::FromPrimitivePushforwardPrep,
    backend::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    return value_and_pushforward!(
        f, ty, prep.pushforward_prep, backend.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::FromPrimitivePushforwardPrep,
    backend::AutoForwardFromPrimitive,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    return value_and_pushforward!(
        f!, y, ty, prep.pushforward_prep, backend.backend, x, tx, contexts...
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
    backend::AutoReverseFromPrimitive{inplace}, dimension::Integer
) where {inplace}
    return AutoReverseFromPrimitive(
        threshold_batchsize(backend.backend, dimension); inplace
    )
end

struct FromPrimitivePullbackPrep{SIG,E<:PullbackPrep} <: PullbackPrep{SIG}
    _sig::Val{SIG}
    pullback_prep::E
end

function prepare_pullback(
    strict::Val,
    f::F,
    backend::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C};
) where {F,C}
    _sig = signature(f, backend, x, ty, contexts...; strict)
    primitive_prep = prepare_pullback(strict, f, backend.backend, x, ty, contexts...)
    return FromPrimitivePullbackPrep(_sig, primitive_prep)
end

function prepare_pullback(
    strict::Val,
    f!::F,
    y,
    backend::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C};
) where {F,C}
    _sig = signature(f!, y, backend, x, ty, contexts...; strict)
    primitive_prep = prepare_pullback(strict, f!, y, backend.backend, x, ty, contexts...)
    return FromPrimitivePullbackPrep(_sig, primitive_prep)
end

function value_and_pullback(
    f::F,
    prep::FromPrimitivePullbackPrep,
    backend::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, ty, contexts...)
    return value_and_pullback(f, prep.pullback_prep, backend.backend, x, ty, contexts...)
end

function value_and_pullback(
    f!::F,
    y,
    prep::FromPrimitivePullbackPrep,
    backend::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    return value_and_pullback(
        f!, y, prep.pullback_prep, backend.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f::F,
    tx::NTuple,
    prep::FromPrimitivePullbackPrep,
    backend::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, ty, contexts...)
    return value_and_pullback!(
        f, tx, prep.pullback_prep, backend.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f!::F,
    y,
    tx::NTuple,
    prep::FromPrimitivePullbackPrep,
    backend::AutoReverseFromPrimitive,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    return value_and_pullback!(
        f!, y, tx, prep.pullback_prep, backend.backend, x, ty, contexts...
    )
end
