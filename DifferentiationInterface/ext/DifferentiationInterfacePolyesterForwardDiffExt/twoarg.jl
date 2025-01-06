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

struct PolyesterForwardDiffTwoArgJacobianPrep{chunksize} <: DI.JacobianPrep
    chunk::Chunk{chunksize}
end

function DI.prepare_jacobian(
    f!, y, ::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{DI.Context,C}
) where {chunksize,C}
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    return PolyesterForwardDiffTwoArgJacobianPrep(chunk)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    ::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{DI.Context,C},
) where {K,C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(fc!, y, jac, x, prep.chunk)
    fc!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    ::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{DI.Context,C},
) where {K,C}
    fc! = DI.with_contexts(f!, contexts...)
    threaded_jacobian!(fc!, y, jac, x, prep.chunk)
    fc!(y, x)
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(fc!, y, jac, x, prep.chunk)
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    threaded_jacobian!(fc!, y, jac, x, prep.chunk)
    return jac
end
