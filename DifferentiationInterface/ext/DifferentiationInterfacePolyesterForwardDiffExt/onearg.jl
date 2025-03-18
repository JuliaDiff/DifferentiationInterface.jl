
## Pushforward

struct PolyesterForwardDiffOneArgPushforwardPrep{SIG,P} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    single_threaded_prep::P
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    single_threaded_prep = DI.prepare_pushforward_nokwarg(
        strict, f, single_threaded(backend), x, tx, contexts...
    )
    return PolyesterForwardDiffOneArgPushforwardPrep(_sig, single_threaded_prep)
end

function DI.value_and_pushforward(
    f,
    prep::PolyesterForwardDiffOneArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward(
        f, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::PolyesterForwardDiffOneArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward!(
        f, ty, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f,
    prep::PolyesterForwardDiffOneArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.pushforward(
        f, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::PolyesterForwardDiffOneArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.pushforward!(
        f, ty, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

## Derivative

struct PolyesterForwardDiffOneArgDerivativePrep{SIG,P} <: DI.DerivativePrep{SIG}
    _sig::Val{SIG}
    single_threaded_prep::P
end

function DI.prepare_derivative_nokwarg(
    strict::Val, f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    single_threaded_prep = DI.prepare_derivative_nokwarg(
        strict, f, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffOneArgDerivativePrep(_sig, single_threaded_prep)
end

function DI.value_and_derivative(
    f,
    prep::PolyesterForwardDiffOneArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_and_derivative(
        f, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_and_derivative!(
    f,
    der,
    prep::PolyesterForwardDiffOneArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_and_derivative!(
        f, der, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.derivative(
    f,
    prep::PolyesterForwardDiffOneArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.derivative(
        f, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.derivative!(
    f,
    der,
    prep::PolyesterForwardDiffOneArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.derivative!(
        f, der, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

## Gradient

struct PolyesterForwardDiffGradientPrep{SIG,chunksize,P} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    chunk::Chunk{chunksize}
    single_threaded_prep::P
end

function DI.prepare_gradient_nokwarg(
    strict::Val,
    f,
    backend::AutoPolyesterForwardDiff{chunksize},
    x,
    contexts::Vararg{DI.Context,C};
) where {chunksize,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    single_threaded_prep = DI.prepare_gradient_nokwarg(
        strict, f, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffGradientPrep(_sig, chunk, single_threaded_prep)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
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
    DI.check_prep(f, prep, backend, x, contexts...)
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
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_and_gradient!(f, similar(x), prep, backend, x, contexts...)
end

function DI.gradient(
    f,
    prep::PolyesterForwardDiffGradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.gradient!(f, similar(x), prep, backend, x, contexts...)
end

## Jacobian

struct PolyesterForwardDiffOneArgJacobianPrep{SIG,chunksize,P} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    chunk::Chunk{chunksize}
    single_threaded_prep::P
end

function DI.prepare_jacobian_nokwarg(
    strict::Val,
    f,
    backend::AutoPolyesterForwardDiff{chunksize},
    x,
    contexts::Vararg{DI.Context,C};
) where {chunksize,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    single_threaded_prep = DI.prepare_jacobian_nokwarg(
        strict, f, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffOneArgJacobianPrep(_sig, chunk, single_threaded_prep)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::PolyesterForwardDiffOneArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
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
    DI.check_prep(f, prep, backend, x, contexts...)
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
    DI.check_prep(f, prep, backend, x, contexts...)
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
    DI.check_prep(f, prep, backend, x, contexts...)
    y = f(x, map(DI.unwrap, contexts)...)
    jac = similar(y, length(y), length(x))
    return DI.jacobian!(f, jac, prep, backend, x, contexts...)
end

## Hessian

struct PolyesterForwardDiffHessianPrep{SIG,P} <: DI.HessianPrep{SIG}
    _sig::Val{SIG}
    single_threaded_prep::P
end

function DI.prepare_hessian_nokwarg(
    strict::Val, f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    single_threaded_prep = DI.prepare_hessian_nokwarg(
        strict, f, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffHessianPrep(_sig, single_threaded_prep)
end

function DI.hessian(
    f,
    prep::PolyesterForwardDiffHessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.hessian(
        f, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.hessian!(
    f,
    hess,
    prep::PolyesterForwardDiffHessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.hessian!(
        f, hess, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_gradient_and_hessian(
    f,
    prep::PolyesterForwardDiffHessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_gradient_and_hessian(
        f, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::PolyesterForwardDiffHessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_gradient_and_hessian!(
        f, grad, hess, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

## Second derivative

struct PolyesterForwardDiffOneArgSecondDerivativePrep{SIG,P} <: DI.SecondDerivativePrep{SIG}
    _sig::Val{SIG}
    single_threaded_prep::P
end

function DI.prepare_second_derivative_nokwarg(
    strict::Val, f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    single_threaded_prep = DI.prepare_second_derivative_nokwarg(
        strict, f, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffOneArgSecondDerivativePrep(_sig, single_threaded_prep)
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::PolyesterForwardDiffOneArgSecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_derivative_and_second_derivative(
        f, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    prep::PolyesterForwardDiffOneArgSecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.value_derivative_and_second_derivative!(
        f, der, der2, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.second_derivative(
    f,
    prep::PolyesterForwardDiffOneArgSecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.second_derivative(
        f, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.second_derivative!(
    f,
    der2,
    prep::PolyesterForwardDiffOneArgSecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return DI.second_derivative!(
        f, der2, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end
