struct MooncakeOneArgPullbackPrep{Tcache,DY} <: DI.PullbackPrep
    cache::Tcache
    dy_righttype::DY
end

function DI.prepare_pullback(
    f, backend::AutoMooncake, x, ty::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    config = get_config(backend)
    cache = prepare_pullback_cache(
        f, x, map(DI.unwrap, contexts)...; config.debug_mode, config.silence_debug_messages
    )
    y = f(x, map(DI.unwrap, contexts)...)
    dy_righttype = zero_tangent(y)
    prep = MooncakeOneArgPullbackPrep(cache, dy_righttype)
    DI.value_and_pullback(f, prep, backend, x, ty, contexts...)
    return prep
end

function DI.value_and_pullback(
    f::F,
    prep::MooncakeOneArgPullbackPrep{Y},
    ::AutoMooncake,
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,Y,C}
    dy = only(ty)
    dy_righttype = dy isa tangent_type(Y) ? dy : copyto!!(prep.dy_righttype, dy)
    new_y, (_, new_dx) = Mooncake.value_and_pullback!!(
        prep.cache, dy_righttype, f, x, map(DI.unwrap, contexts)...
    )
    return new_y, (deepcopy(new_dx),)
end

function DI.value_and_pullback!(
    f,
    tx::NTuple{1},
    prep::MooncakeOneArgPullbackPrep{Y},
    backend::AutoMooncake,
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {Y,C}
    y, (new_dx,) = DI.value_and_pullback(f, prep, backend, x, ty, contexts...)
    copyto!(only(tx), new_dx)
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    ys_and_tx = map(ty) do dy
        y, tx = DI.value_and_pullback(f, prep, backend, x, (dy,), contexts...)
        y, only(tx)
    end
    y = first(ys_and_tx[1])
    tx = last.(ys_and_tx)
    return y, tx
end

function DI.value_and_pullback!(
    f,
    tx::NTuple,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    ys = map(tx, ty) do dx, dy
        y, _ = DI.value_and_pullback!(f, (dx,), prep, backend, x, (dy,), contexts...)
        y
    end
    y = ys[1]
    return y, tx
end

function DI.pullback(
    f,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function DI.pullback!(
    f,
    tx::NTuple,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)[2]
end
