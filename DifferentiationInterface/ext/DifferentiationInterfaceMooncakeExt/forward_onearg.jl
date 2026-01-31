## Pushforward

struct MooncakeOneArgPushforwardPrep{SIG, Tcache, FT, CT} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    df::FT
    context_tangents::CT
end

function DI.prepare_pushforward_nokwarg(
        strict::Val,
        f::F,
        backend::AutoMooncakeForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    config = get_config(backend)
    cache = prepare_derivative_cache(f, x, map(DI.unwrap, contexts)...; config)
    df = zero_tangent_or_primal(f, backend)
    context_tangents = map(zero_tangent_unwrap, contexts)
    prep = MooncakeOneArgPushforwardPrep(_sig, cache, df, context_tangents)
    return prep
end

function DI.value_and_pushforward(
        f::F,
        prep::MooncakeOneArgPushforwardPrep,
        backend::AutoMooncakeForward,
        x::X,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C, X}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ys_and_ty = map(tx) do dx
        y_and_dy = value_and_derivative!!(
            prep.cache,
            (f, prep.df),
            (x, dx),
            map(first_unwrap, contexts, prep.context_tangents)...,
        )
        y = first(y_and_dy)
        dy = _copy_output(last(y_and_dy))
        return y, dy
    end
    y = _copy_output(first(ys_and_ty[1]))
    ty = map(last, ys_and_ty)
    return y, ty
end

function DI.pushforward(
        f::F,
        prep::MooncakeOneArgPushforwardPrep,
        backend::AutoMooncakeForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)[2]
end

function DI.value_and_pushforward!(
        f::F,
        ty::NTuple,
        prep::MooncakeOneArgPushforwardPrep,
        backend::AutoMooncakeForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    y, new_ty = DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function DI.pushforward!(
        f::F,
        ty::NTuple,
        prep::MooncakeOneArgPushforwardPrep,
        backend::AutoMooncakeForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    DI.value_and_pushforward!(f, ty, prep, backend, x, tx, contexts...)
    return ty
end
