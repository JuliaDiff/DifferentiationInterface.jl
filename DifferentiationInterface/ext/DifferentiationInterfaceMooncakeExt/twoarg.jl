struct MooncakeTwoArgPullbackPrep{SIG, Tcache, DY, N} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    dy_backup::DY
    args_to_zero::NTuple{N, Bool}
end

function DI.prepare_pullback_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoMooncake,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f!, y, backend, x, ty, contexts...; strict)
    config = get_config(backend)
    cache = prepare_pullback_cache(
        call_and_return,
        f!,
        y,
        x,
        map(DI.unwrap, contexts)...;
        config,
    )
    dy_backup = zero_tangent_or_primal(y, backend)
    contexts_tup_false = map(_ -> false, contexts)
    args_to_zero = (
        false,  # call_and_return
        false,  # f!
        false,  # y
        true,  # x
        contexts_tup_false...,
    )
    prep = MooncakeTwoArgPullbackPrep(
        _sig, cache, dy_backup, args_to_zero
    )
    return prep
end

function DI.value_and_pullback(
        f!::F,
        y,
        prep::MooncakeTwoArgPullbackPrep,
        backend::AutoMooncake,
        x,
        ty::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    dy = only(ty)
    # Prepare cotangent to add after the forward pass.
    dy_backup = copyto!(prep.dy_backup, dy)
    # Run the reverse-pass and return the results.
    y_after, (_, _, _, dx) = value_and_pullback!!(
        prep.cache,
        dy_backup,
        call_and_return,
        f!,
        y,
        x,
        map(DI.unwrap, contexts)...;
        prep.args_to_zero
    )
    copyto!(y, y_after)
    return y, (_copy_output(dx),)
end

function DI.value_and_pullback(
        f!::F,
        y,
        prep::MooncakeTwoArgPullbackPrep,
        backend::AutoMooncake,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    tx = map(ty) do dy
        dy_backup = copyto!(prep.dy_backup, dy)
        y_after, (_, _, _, dx) = value_and_pullback!!(
            prep.cache,
            dy_backup,
            call_and_return,
            f!,
            y,
            x,
            map(DI.unwrap, contexts)...;
            prep.args_to_zero
        )
        copyto!(y, y_after)
        _copy_output(dx)
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
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
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
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
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
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)[2]
end
