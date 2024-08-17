abstract type FromPrimitive <: AbstractADType end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
twoarg_support(fromprim::FromPrimitive) = twoarg_support(fromprim.backend)

function pick_batchsize(fromprim::FromPrimitive, dimension::Integer)
    return pick_batchsize(fromprim.backend, dimension)
end

## Forward

struct AutoForwardFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoForwardFromPrimitive) = ADTypes.ForwardMode()

struct FromPrimitivePushforwardExtras{E<:PushforwardExtras} <: PushforwardExtras
    pushforward_extras::E
end

function prepare_pushforward(f, fromprim::AutoForwardFromPrimitive, x, tx)
    return FromPrimitivePushforwardExtras(prepare_pushforward(f, fromprim.backend, x, tx))
end

function prepare_pushforward(f!, y, fromprim::AutoForwardFromPrimitive, x, tx)
    return FromPrimitivePushforwardExtras(
        prepare_pushforward(f!, y, fromprim.backend, x, tx)
    )
end

function value_and_pushforward(
    f, fromprim::AutoForwardFromPrimitive, x, tx, extras::FromPrimitivePushforwardExtras
)
    return value_and_pushforward(f, fromprim.backend, x, tx, extras.pushforward_extras)
end

function value_and_pushforward(
    f!, y, fromprim::AutoForwardFromPrimitive, x, tx, extras::FromPrimitivePushforwardExtras
)
    return value_and_pushforward(f!, y, fromprim.backend, x, tx, extras.pushforward_extras)
end

function value_and_pushforward!(
    f, ty, fromprim::AutoForwardFromPrimitive, x, tx, extras::FromPrimitivePushforwardExtras
)
    return value_and_pushforward!(f, ty, fromprim.backend, x, tx, extras.pushforward_extras)
end

function value_and_pushforward!(
    f!,
    y,
    ty,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx,
    extras::FromPrimitivePushforwardExtras,
)
    return value_and_pushforward!(
        f!, y, ty, fromprim.backend, x, tx, extras.pushforward_extras
    )
end

## Reverse

struct AutoReverseFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

struct FromPrimitivePullbackExtras{E<:PullbackExtras} <: PullbackExtras
    pullback_extras::E
end

function prepare_pullback(f, fromprim::AutoReverseFromPrimitive, x, ty)
    return FromPrimitivePullbackExtras(prepare_pullback(f, fromprim.backend, x, ty))
end

function prepare_pullback(f!, y, fromprim::AutoReverseFromPrimitive, x, ty)
    return FromPrimitivePullbackExtras(prepare_pullback(f!, y, fromprim.backend, x, ty))
end

function value_and_pullback(
    f, fromprim::AutoReverseFromPrimitive, x, ty, extras::FromPrimitivePullbackExtras
)
    return value_and_pullback(f, fromprim.backend, x, ty, extras.pullback_extras)
end

function value_and_pullback(
    f!, y, fromprim::AutoReverseFromPrimitive, x, ty, extras::FromPrimitivePullbackExtras
)
    return value_and_pullback(f!, y, fromprim.backend, x, ty, extras.pullback_extras)
end

function value_and_pullback!(
    f, tx, fromprim::AutoReverseFromPrimitive, x, ty, extras::FromPrimitivePullbackExtras
)
    return value_and_pullback!(f, tx, fromprim.backend, x, ty, extras.pullback_extras)
end

function value_and_pullback!(
    f!,
    y,
    tx,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty,
    extras::FromPrimitivePullbackExtras,
)
    return value_and_pullback!(f!, y, tx, fromprim.backend, x, ty, extras.pullback_extras)
end
