
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

struct PolyesterForwardDiffGradientPrep{chunksize,P} <: DI.GradientPrep
    chunk::Chunk{chunksize}
    single_threaded_prep::P
end

function DI.prepare_gradient(
    f, backend::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{DI.Context,C}
) where {chunksize,C}
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    single_threaded_prep = DI.prepare_gradient(f, single_threaded(backend), x, contexts...)
    return PolyesterForwardDiffGradientPrep(chunk, single_threaded_prep)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc = DI.with_contexts(f, contexts...)
        threaded_gradient!(fc, grad, x, prep.chunk)
        return fc(x), grad
    else
        # TODO: optimize
        return DI.value_and_gradient!(
            f, grad, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.gradient!(
    f,
    grad,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc = DI.with_contexts(f, contexts...)
        threaded_gradient!(fc, grad, x, prep.chunk)
        return grad
    else
        # TODO: optimize
        return DI.gradient!(
            f, grad, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
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

struct PolyesterForwardDiffOneArgJacobianPrep{chunksize,P} <: DI.JacobianPrep
    chunk::Chunk{chunksize}
    single_threaded_prep::P
end

function DI.prepare_jacobian(
    f, backend::AutoPolyesterForwardDiff{chunksize}, x, contexts::Vararg{DI.Context,C}
) where {chunksize,C}
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    single_threaded_prep = DI.prepare_jacobian(f, single_threaded(backend), x, contexts...)
    return PolyesterForwardDiffOneArgJacobianPrep(chunk, single_threaded_prep)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc = DI.with_contexts(f, contexts...)
        return fc(x), threaded_jacobian!(fc, jac, x, prep.chunk)
    else
        return DI.value_and_jacobian!(
            f, jac, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.jacobian!(
    f,
    jac,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc = DI.with_contexts(f, contexts...)
        return threaded_jacobian!(fc, jac, x, prep.chunk)
    else
        return DI.jacobian!(
            f, jac, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.value_and_jacobian(
    f,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    y = f(x, map(DI.unwrap, contexts)...)
    jac = similar(y, length(y), length(x))
    return DI.value_and_jacobian!(f, jac, prep, backend, x, contexts...)
end

function DI.jacobian(
    f,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    y = f(x, map(DI.unwrap, contexts)...)
    jac = similar(y, length(y), length(x))
    return DI.jacobian!(f, jac, prep, backend, x, contexts...)
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

## HVP

#=
function DI.prepare_hvp(
    f, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.prepare_hvp(
        f, DI.SecondOrder(single_threaded(backend), backend), x, tx, contexts...
    )
end

function DI.hvp(
    f,
    prep::DI.ForwardOverAnythingHVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.hvp(
        f, prep, DI.SecondOrder(single_threaded(backend), backend), x, tx, contexts...
    )
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::DI.ForwardOverAnythingHVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.hvp!(
        f, tg, prep, DI.SecondOrder(single_threaded(backend), backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp(
    f,
    prep::DI.ForwardOverAnythingHVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.gradient_and_hvp(
        f, prep, DI.SecondOrder(single_threaded(backend), backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp!(
    f,
    grad,
    tg::NTuple,
    prep::DI.ForwardOverAnythingHVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return DI.gradient_and_hvp!(
        f,
        grad,
        tg,
        prep,
        DI.SecondOrder(single_threaded(backend), backend),
        x,
        tx,
        contexts...,
    )
end
=#

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
