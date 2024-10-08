
## Pushforward

function DI.prepare_pushforward(
    f, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_pushforward(f, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward(
    f,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pushforward(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pushforward!(
        f, ty, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.pushforward(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.pushforward!(f, ty, prep, single_threaded(backend), x, tx, contexts...)
end

## Derivative

function DI.prepare_derivative(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_derivative(f, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative(
    f,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative!(
    f,
    der,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_derivative!(f, der, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative(
    f,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative!(
    f,
    der,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.derivative!(f, der, prep, single_threaded(backend), x, contexts...)
end

## Gradient

struct PolyesterForwardDiffGradientPrep{chunksize} <: GradientPrep
    chunk::Chunk{chunksize}
end

function DI.prepare_gradient(
    f, ::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{Context,C}
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
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    threaded_gradient!(fc, grad, x, prep.chunk)
    return fc(x), grad
end

function DI.gradient!(
    f,
    grad,
    prep::PolyesterForwardDiffGradientPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    threaded_gradient!(fc, grad, x, prep.chunk)
    return grad
end

function DI.value_and_gradient(
    f,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_gradient!(f, similar(x), prep, backend, x, contexts...)
end

function DI.gradient(
    f,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.gradient!(f, similar(x), prep, backend, x, contexts...)
end

## Jacobian

struct PolyesterForwardDiffOneArgJacobianPrep{chunksize} <: JacobianPrep
    chunk::Chunk{chunksize}
end

function DI.prepare_jacobian(
    f, ::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{Context,C}
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
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return fc(x), threaded_jacobian!(fc, jac, x, prep.chunk)
end

function DI.jacobian!(
    f,
    jac,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    ::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return threaded_jacobian!(fc, jac, x, prep.chunk)
end

function DI.value_and_jacobian(
    f,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    y = f(x, map(unwrap, contexts)...)
    return DI.value_and_jacobian!(
        f, similar(y, length(y), length(x)), prep, backend, x, contexts...
    )
end

function DI.jacobian(
    f,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    y = f(x, map(unwrap, contexts)...)
    return DI.jacobian!(f, similar(y, length(y), length(x)), prep, backend, x, contexts...)
end

## Hessian

function DI.prepare_hessian(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_hessian(f, single_threaded(backend), x, contexts...)
end

function DI.hessian(
    f, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.hessian(f, prep, single_threaded(backend), x, contexts...)
end

function DI.hessian!(
    f,
    hess,
    prep::HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.hessian!(f, hess, prep, single_threaded(backend), x, contexts...)
end

function DI.value_gradient_and_hessian(
    f, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.value_gradient_and_hessian(f, prep, single_threaded(backend), x, contexts...)
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_gradient_and_hessian!(
        f, grad, hess, prep, single_threaded(backend), x, contexts...
    )
end

## HVP

function DI.prepare_hvp(
    f, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_hvp(f, single_threaded(backend), x, tx, contexts...)
end

function DI.hvp(
    f,
    prep::HVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.hvp(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::HVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.hvp!(f, tg, prep, single_threaded(backend), x, tx, contexts...)
end

## Second derivative

function DI.prepare_second_derivative(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_second_derivative(f, single_threaded(backend), x, contexts...)
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_derivative_and_second_derivative(
        f, prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_derivative_and_second_derivative!(
        f, der, der2, prep, single_threaded(backend), x, contexts...
    )
end

function DI.second_derivative(
    f,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.second_derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.second_derivative!(
    f,
    der2,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.second_derivative!(f, der2, prep, single_threaded(backend), x, contexts...)
end
