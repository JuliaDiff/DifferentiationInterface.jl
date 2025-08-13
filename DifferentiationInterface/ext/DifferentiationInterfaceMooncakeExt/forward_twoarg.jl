## Pushforward

struct MooncakeTwoArgPushforwardPrep{SIG,Tcache,DX,DY} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    dx_righttype::DX
    dy_righttype::DY
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f!::F,
    y,
    backend::AutoMooncakeForward,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    config = get_config(backend)
    cache = prepare_derivative_cache(
        f!,
        y,
        x,
        map(DI.unwrap, contexts)...;
        config.debug_mode,
        config.silence_debug_messages,
    )
    dx_righttype = zero_tangent(x)
    dy_righttype = zero_tangent(y)
    prep = MooncakeTwoArgPushforwardPrep(_sig, cache, dx_righttype, dy_righttype)
    return prep
end

function DI.value_and_pushforward(
    f!::F,
    y,
    prep::MooncakeTwoArgPushforwardPrep,
    backend::AutoMooncakeForward,
    x::X,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C,X}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = map(tx) do dx
        dx_righttype =
            dx isa tangent_type(X) ? dx : _copy_to_output!!(prep.dx_righttype, dx)
        y_dual = zero_dual(y)
        value_and_derivative!!(
            prep.cache,
            zero_dual(f!),
            y_dual,
            Dual(x, dx_righttype),
            map(zero_dual ∘ DI.unwrap, contexts)...,
        )
        dy = _copy_output(tangent(y_dual))
        return dy
    end
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    prep::MooncakeTwoArgPushforwardPrep,
    backend::AutoMooncakeForward,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)[2]
end

function DI.value_and_pushforward!(
    f!::F,
    y::Y,
    ty::NTuple,
    prep::MooncakeTwoArgPushforwardPrep,
    backend::AutoMooncakeForward,
    x::X,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C,X,Y}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    foreach(tx, ty) do dx, dy
        dx_righttype =
            dx isa tangent_type(X) ? dx : _copy_to_output!!(prep.dx_righttype, dx)
        dy_righttype =
            dy isa tangent_type(Y) ? dy : _copy_to_output!!(prep.dy_righttype, dy)
        value_and_derivative!!(
            prep.cache,
            zero_dual(f!),
            Dual(y, dy_righttype),
            Dual(x, dx_righttype),
            map(zero_dual ∘ DI.unwrap, contexts)...,
        )
        dy === dy_righttype || copyto!(dy, dy_righttype)
    end
    return y, ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::MooncakeTwoArgPushforwardPrep,
    backend::AutoMooncakeForward,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    return ty
end
