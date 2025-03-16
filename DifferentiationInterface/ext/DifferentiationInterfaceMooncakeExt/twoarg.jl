struct MooncakeTwoArgPullbackPrep{SIG,Tcache,DY,F} <: DI.PullbackPrep{SIG}
    _sig::Type{SIG}
    cache::Tcache
    dy_righttype::DY
    target_function::F
end

function DI.prepare_pullback(
    f!::F,
    y,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {F,C}
    SIG = DI.signature(f!, y, backend, x, ty, contexts...; strict)
    target_function = function (f!, y, x, contexts...)
        f!(y, x, contexts...)
        return y
    end
    config = get_config(backend)
    cache = prepare_pullback_cache(
        target_function,
        f!,
        y,
        x,
        map(DI.unwrap, contexts)...;
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    dy_righttype_after = zero_tangent(y)
    prep = MooncakeTwoArgPullbackPrep(SIG, cache, dy_righttype_after, target_function)
    DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
    return prep
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    # Prepare cotangent to add after the forward pass.
    dy = only(ty)
    dy_righttype_after = copyto!(prep.dy_righttype, dy)

    # Run the reverse-pass and return the results.
    contexts = map(DI.unwrap, contexts)
    y_after, (_, _, _, dx) = value_and_pullback!!(
        prep.cache, dy_righttype_after, prep.target_function, f!, y, x, contexts...
    )
    copyto!(y, y_after)
    return y, (mycopy(dx),)
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    tx = map(ty) do dy
        _, tx = DI.value_and_pullback(f!, y, prep, backend, x, (dy,), contexts...)
        only(tx)
    end
    return y, tx
end

function DI.value_and_pullback!(
    f!::F,
    y,
    tx::NTuple,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    _, new_tx = DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
    foreach(copyto!, tx, new_tx)
    return y, tx
end

function DI.pullback(
    f!::F,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)[2]
end

function DI.pullback!(
    f!::F,
    y,
    tx::NTuple,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)[2]
end
