abstract type FromPrimitive <: AbstractADType end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
inplace_support(fromprim::FromPrimitive) = inplace_support(fromprim.backend)
pushforward_performance(fromprim::FromPrimitive) = pushforward_performance(fromprim.backend)
pullback_performance(fromprim::FromPrimitive) = pullback_performance(fromprim.backend)

basis(fromprim::FromPrimitive, x::AbstractArray, i) = basis(fromprim.backend, x, i)

function pick_batchsize(fromprim::FromPrimitive, x_or_N::Union{AbstractArray,Integer})
    return pick_batchsize(fromprim.backend, x_or_N)
end

struct AutoReverseFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

function threshold_batchsize(fromprim::AutoReverseFromPrimitive, dimension::Integer)
    return AutoReverseFromPrimitive(threshold_batchsize(fromprim.backend, dimension))
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
