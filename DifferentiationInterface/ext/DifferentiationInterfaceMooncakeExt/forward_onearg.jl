## Pushforward

struct MooncakeOneArgPushforwardPrep{SIG,Tcache,DX} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    dx_righttype::DX
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f::F,
    backend::AutoMooncakeForward,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    config = get_config(backend)
    cache = prepare_derivative_cache(
        f, x, map(DI.unwrap, contexts)...; config.debug_mode, config.silence_debug_messages
    )
    dx_righttype = zero_tangent(x)
    prep = MooncakeOneArgPushforwardPrep(_sig, cache, dx_righttype)
    return prep
end

function DI.value_and_pushforward(
    f::F,
    prep::MooncakeOneArgPushforwardPrep,
    backend::AutoMooncakeForward,
    x::X,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C,X}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ys_and_ty = map(tx) do dx
        dx_righttype =
            dx isa tangent_type(X) ? dx : _copy_to_output!!(prep.dx_righttype, dx)
        y_dual = value_and_derivative!!(
            prep.cache,
            zero_dual(f),
            Dual(x, dx_righttype),
            map(zero_dual ∘ DI.unwrap, contexts)...,
        )
        y = primal(y_dual)
        dy = _copy_output(tangent(y_dual))
        return y, dy
    end
    y = first(ys_and_ty[1])
    ty = last.(ys_and_ty)
    return y, ty
end

function DI.pushforward(
    f::F,
    prep::MooncakeOneArgPushforwardPrep,
    backend::AutoMooncakeForward,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C}
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
    contexts::Vararg{DI.Context,C};
) where {F,C}
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
    contexts::Vararg{DI.Context,C};
) where {F,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    DI.value_and_pushforward!(f, ty, prep, backend, x, tx, contexts...)
    return ty
end
