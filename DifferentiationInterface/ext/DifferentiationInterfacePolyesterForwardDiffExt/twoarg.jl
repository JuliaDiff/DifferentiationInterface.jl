## Pushforward

function DI.prepare_pushforward(
    f!, y, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_pushforward(f!, y, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_pushforward(
        f!, y, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_pushforward!(
        f!, y, ty, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f!,
    y,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.pushforward(f!, y, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.pushforward!(f!, y, ty, prep, single_threaded(backend), x, tx, contexts...)
end

## Derivative

function DI.prepare_derivative(
    f!, y, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_derivative(f!, y, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative(
    f!,
    y,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_derivative(f!, y, prep, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_derivative!(
        f!, y, der, prep, single_threaded(backend), x, contexts...
    )
end

function DI.derivative(
    f!,
    y,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.derivative(f!, y, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.derivative!(f!, y, der, prep, single_threaded(backend), x, contexts...)
end

## Jacobian

struct PolyesterForwardDiffTwoArgJacobianPrep{chunksize,P} <: DI.JacobianPrep
    chunk::Chunk{chunksize}
    single_threaded_prep::P
end

function DI.prepare_jacobian(
    f!, y, backend::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{DI.Context,C}
) where {chunksize,C}
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    single_threaded_prep = DI.prepare_jacobian(
        f!, y, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffTwoArgJacobianPrep(chunk, single_threaded_prep)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{DI.Context,C},
) where {K,C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        jac = similar(y, length(y), length(x))
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        fc!(y, x)
        return y, jac
    else
        return DI.value_and_jacobian(
            f!, y, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{DI.Context,C},
) where {K,C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        fc!(y, x)
        return y, jac
    else
        return DI.value_and_jacobian!(
            f!, y, jac, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.jacobian(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        jac = similar(y, length(y), length(x))
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        return jac
    else
        return DI.jacobian(
            f!, y, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        return jac
    else
        return DI.jacobian!(
            f!, y, jac, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end
