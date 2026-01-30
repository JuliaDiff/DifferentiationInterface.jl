## Pushforward

struct MooncakeTwoArgPushforwardPrep{SIG, Tcache, FT, CT} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    df!::FT
    context_tangents::CT
end

function DI.prepare_pushforward_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoMooncakeForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    config = get_config(backend)
    cache = prepare_derivative_cache(
        call_and_return,
        f!,
        y,
        x,
        map(DI.unwrap, contexts)...;
        config
    )
    df! = zero_tangent(f!)
    context_tangents = map(zero_tangent_unwrap, contexts)
    prep = MooncakeTwoArgPushforwardPrep(_sig, cache, df!, context_tangents)
    return prep
end

function DI.value_and_pushforward(
        f!::F,
        y,
        prep::MooncakeTwoArgPushforwardPrep,
        backend::AutoMooncakeForward,
        x::X,
        tx::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C, X}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = map(tx) do dx
        dy = zero_tangent(y)  # TODO: remove allocation?
        _, new_dy = value_and_derivative!!(
            prep.cache,
            (call_and_return, zero_tangent(call_and_return)),
            (f!, prep.df!),
            (y, dy),
            (x, dx),
            map(first_unwrap, contexts, prep.context_tangents)...,
        )
        return _copy_output(new_dy)
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
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
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
        contexts::Vararg{DI.Context, C}
    ) where {F, C, X, Y}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    foreach(tx, ty) do dx, dy
        _, new_dy = value_and_derivative!!(
            prep.cache,
            (call_and_return, zero_tangent(call_and_return)),
            (f!, prep.df!),
            (y, dy),
            (x, dx),
            map(first_unwrap, contexts, prep.context_tangents)...,
        )
        copyto!(dy, new_dy)
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
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    return ty
end
