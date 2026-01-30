## Pullback

struct MooncakeOneArgPullbackPrep{SIG, Tcache, N} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    args_to_zero::NTuple{N, Bool}
end

function DI.prepare_pullback_nokwarg(
        strict::Val, f::F, backend::AutoMooncake, x, ty::NTuple, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    config = get_config(backend)
    cache = prepare_pullback_cache(
        f, x, map(DI.unwrap, contexts)...; config
    )
    contexts_tup_false = map(_ -> false, contexts)
    args_to_zero = (
        false,  # f
        true,  # x
        contexts_tup_false...,
    )
    prep = MooncakeOneArgPullbackPrep(_sig, cache, args_to_zero)
    return prep
end

function DI.value_and_pullback(
        f::F,
        prep::MooncakeOneArgPullbackPrep{Y},
        backend::AutoMooncake,
        x,
        ty::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, Y, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    dy = only(ty)
    new_y, (_, new_dx) = value_and_pullback!!(
        prep.cache, dy, f, x, map(DI.unwrap, contexts)...; prep.args_to_zero
    )
    return new_y, (_copy_output(new_dx),)
end

function DI.value_and_pullback(
        f::F,
        prep::MooncakeOneArgPullbackPrep{Y},
        backend::AutoMooncake,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, Y, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    ys_and_tx = map(ty) do dy
        y, (_, new_dx) = value_and_pullback!!(
            prep.cache, dy, f, x, map(DI.unwrap, contexts)...; prep.args_to_zero
        )
        y, _copy_output(new_dx)
    end
    y = first(ys_and_tx[1])
    tx = map(last, ys_and_tx)
    return y, tx
end

function DI.value_and_pullback!(
        f::F,
        tx::NTuple,
        prep::MooncakeOneArgPullbackPrep,
        backend::AutoMooncake,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    y, new_tx = DI.value_and_pullback(f, prep, backend, x, ty, contexts...)
    foreach(copyto!, tx, new_tx)
    return y, tx
end

function DI.pullback(
        f::F,
        prep::MooncakeOneArgPullbackPrep,
        backend::AutoMooncake,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function DI.pullback!(
        f::F,
        tx::NTuple,
        prep::MooncakeOneArgPullbackPrep,
        backend::AutoMooncake,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)[2]
end

## Gradient

struct MooncakeGradientPrep{SIG, Tcache, N} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
    args_to_zero::NTuple{N, Bool}
end

function DI.prepare_gradient_nokwarg(
        strict::Val, f::F, backend::AutoMooncake, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    config = get_config(backend)
    cache = prepare_gradient_cache(
        f, x, map(DI.unwrap, contexts)...; config
    )
    contexts_tup_false = map(_ -> false, contexts)
    args_to_zero = (
        false,  # f
        true,  # x
        contexts_tup_false...,
    )
    prep = MooncakeGradientPrep(_sig, cache, args_to_zero)
    return prep
end

function DI.value_and_gradient(
        f::F,
        prep::MooncakeGradientPrep,
        backend::AutoMooncake,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, (_, new_grad) = value_and_gradient!!(
        prep.cache, f, x, map(DI.unwrap, contexts)...;
        prep.args_to_zero
    )
    return y, _copy_output(new_grad)
end

function DI.value_and_gradient!(
        f::F,
        grad,
        prep::MooncakeGradientPrep,
        backend::AutoMooncake,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, (_, new_grad) = value_and_gradient!!(
        prep.cache, f, x, map(DI.unwrap, contexts)...;
        prep.args_to_zero
    )
    copyto!(grad, new_grad)
    return y, grad
end

function DI.gradient(
        f::F,
        prep::MooncakeGradientPrep,
        backend::AutoMooncake,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    _, grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return grad
end

function DI.gradient!(
        f::F,
        grad,
        prep::MooncakeGradientPrep,
        backend::AutoMooncake,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    DI.value_and_gradient!(f, grad, prep, backend, x, contexts...)
    return grad
end
