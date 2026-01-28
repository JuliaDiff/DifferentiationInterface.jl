## Pushforward

struct MooncakeTwoArgPushforwardPrep{SIG, Tcache, DX, DY, FT, CT} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    dx_righttype::DX
    dy_righttype::DY
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
        f!,
        y,
        x,
        map(DI.unwrap, contexts)...;
        config,
    )
    if config.friendly_tangents
        dx_righttype = zero_tangent(x)
        dy_righttype = zero_tangent(y)
    else
        dx_righttype = nothing
        dy_righttype = nothing
    end
    df! = zero_tangent(f!)
    context_tangents = map(zero_tangent_unwrap, contexts)
    prep = MooncakeTwoArgPushforwardPrep(_sig, cache, dx_righttype, dy_righttype, df!, context_tangents)
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
        dx_righttype =
            isnothing(prep.dx_righttype) ? dx : primal_to_tangent!!(prep.dx_righttype, dx)
        y_dual = zero_dual(y)
        value_and_derivative!!(
            prep.cache,
            Dual(f!, prep.df!),
            y_dual,
            Dual(x, dx_righttype),
            map(Dual_unwrap, contexts, prep.context_tangents)...,
        )
        if isnothing(prep.dx_righttype)
            dy = _copy_output(tangent(y_dual))
        else
            dy = tangent_to_primal!!(_copy_output(y), tangent(y_dual))
        end
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
        dx_righttype =
            isnothing(prep.dx_righttype) ? dx : primal_to_tangent!!(prep.dx_righttype, dx)
        dy_righttype =
            isnothing(prep.dy_righttype) ? dy : primal_to_tangent!!(prep.dy_righttype, dy)
        value_and_derivative!!(
            prep.cache,
            Dual(f!, prep.df!),
            Dual(y, dy_righttype),
            Dual(x, dx_righttype),
            map(Dual_unwrap, contexts, prep.context_tangents)...,
        )
        isnothing(prep.dy_righttype) || tangent_to_primal!!(dy, dy_righttype)
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
