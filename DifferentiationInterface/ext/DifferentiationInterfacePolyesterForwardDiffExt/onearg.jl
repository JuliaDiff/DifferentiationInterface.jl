
## Pushforward

function DI.prepare_pushforward(
    f, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_pushforward(f, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward(
    f,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_pushforward(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_pushforward!(
        f, ty, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.pushforward(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::DI.PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.pushforward!(f, ty, prep, single_threaded(backend), x, tx, contexts...)
end

## Derivative

function DI.prepare_derivative(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_derivative(f, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative(
    f,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative!(
    f,
    der,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_derivative!(f, der, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative(
    f,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative!(
    f,
    der,
    prep::DI.DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.derivative!(f, der, prep, single_threaded(backend), x, contexts...)
end

## Gradient

struct PolyesterForwardDiffGradientPrep{chunksize} <: DI.GradientPrep
    chunk::Chunk{chunksize}
end

function DI.prepare_gradient(
    f, ::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{DI.Context,C}
) where {chunksize,C}
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    return PolyesterForwardDiffGradientPrep(chunk)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::PolyesterForwardDiffGradientPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    threaded_gradient!(fc, grad, x, prep.chunk)
    return fc(x), grad
end

function DI.gradient!(
    f,
    grad,
    prep::PolyesterForwardDiffGradientPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    threaded_gradient!(fc, grad, x, prep.chunk)
    return grad
end

function DI.value_and_gradient(
    f,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_and_gradient!(f, similar(x), prep, backend, x, contexts...)
end

function DI.gradient(
    f,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.gradient!(f, similar(x), prep, backend, x, contexts...)
end

## Jacobian

struct PolyesterForwardDiffOneArgJacobianPrep{chunksize} <: DI.JacobianPrep
    chunk::Chunk{chunksize}
end

function DI.prepare_jacobian(
    f, ::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{DI.Context,C}
) where {chunksize,C}
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    return PolyesterForwardDiffOneArgJacobianPrep(chunk)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return fc(x), threaded_jacobian!(fc, jac, x, prep.chunk)
end

function DI.jacobian!(
    f,
    jac,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return threaded_jacobian!(fc, jac, x, prep.chunk)
end

function DI.value_and_jacobian(
    f,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    y = f(x, map(DI.unwrap, contexts)...)
    return DI.value_and_jacobian!(
        f, similar(y, length(y), length(x)), prep, backend, x, contexts...
    )
end

function DI.jacobian(
    f,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    y = f(x, map(DI.unwrap, contexts)...)
    return DI.jacobian!(f, similar(y, length(y), length(x)), prep, backend, x, contexts...)
end

## Hessian

function DI.prepare_hessian(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_hessian(f, single_threaded(backend), x, contexts...)
end

function DI.hessian(
    f,
    prep::DI.HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.hessian(f, prep, single_threaded(backend), x, contexts...)
end

function DI.hessian!(
    f,
    hess,
    prep::DI.HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.hessian!(f, hess, prep, single_threaded(backend), x, contexts...)
end

function DI.value_gradient_and_hessian(
    f,
    prep::DI.HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_gradient_and_hessian(f, prep, single_threaded(backend), x, contexts...)
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::DI.HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_gradient_and_hessian!(
        f, grad, hess, prep, single_threaded(backend), x, contexts...
    )
end

## Second derivative

function DI.prepare_second_derivative(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_second_derivative(f, single_threaded(backend), x, contexts...)
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::DI.SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_derivative_and_second_derivative(
        f, prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    prep::DI.SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.value_derivative_and_second_derivative!(
        f, der, der2, prep, single_threaded(backend), x, contexts...
    )
end

function DI.second_derivative(
    f,
    prep::DI.SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.second_derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.second_derivative!(
    f,
    der2,
    prep::DI.SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.second_derivative!(f, der2, prep, single_threaded(backend), x, contexts...)
end
