## Pushforward

function DI.prepare_pushforward(
    f, ::AutoFiniteDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {C}
    return NoPushforwardPrep()
end

function DI.pushforward(
    f,
    ::NoPushforwardPrep,
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    step(t::Number, dx) = f(x .+ t .* dx, map(unwrap, contexts)...)
    ty = map(tx) do dx
        finite_difference_derivative(Base.Fix2(step, dx), zero(eltype(x)), fdtype(backend))
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    ::NoPushforwardPrep,
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    step(t::Number, dx) = f(x .+ t .* dx, map(unwrap, contexts)...)
    y = f(x, map(unwrap, contexts)...)
    ty = map(tx) do dx
        finite_difference_derivative(
            Base.Fix2(step, dx), zero(eltype(x)), fdtype(backend), eltype(y), y
        )
    end
    return y, ty
end

## Derivative

struct FiniteDiffOneArgDerivativePrep{C} <: DerivativePrep
    cache::C
end

function DI.prepare_derivative(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    cache = if y isa Number
        nothing
    elseif y isa AbstractArray
        df = similar(y)
        cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_NOT_INPLACE)
    end
    return FiniteDiffOneArgDerivativePrep(cache)
end

### Scalar to scalar

function DI.derivative(
    f,
    ::FiniteDiffOneArgDerivativePrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_derivative(fc, x, fdtype(backend))
end

function DI.value_and_derivative(
    f,
    ::FiniteDiffOneArgDerivativePrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    return y, finite_difference_derivative(fc, x, fdtype(backend), eltype(y), y)
end

### Scalar to array

function DI.derivative(
    f,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_gradient(fc, x, prep.cache)
end

function DI.derivative!(
    f,
    der,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_gradient!(der, fc, x, prep.cache)
end

function DI.value_and_derivative(
    f,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    return y, finite_difference_gradient(fc, x, prep.cache)
end

function DI.value_and_derivative!(
    f,
    der,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return fc(x), finite_difference_gradient!(der, fc, x, prep.cache)
end

## Gradient

struct FiniteDiffGradientPrep{C} <: GradientPrep
    cache::C
end

function DI.prepare_gradient(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    df = zero(y) .* x
    cache = GradientCache(df, x, fdtype(backend))
    return FiniteDiffGradientPrep(cache)
end

function DI.gradient(
    f,
    prep::FiniteDiffGradientPrep,
    ::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_gradient(fc, x, prep.cache)
end

function DI.value_and_gradient(
    f,
    prep::FiniteDiffGradientPrep,
    ::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return fc(x), finite_difference_gradient(fc, x, prep.cache)
end

function DI.gradient!(
    f,
    grad,
    prep::FiniteDiffGradientPrep,
    ::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_gradient!(grad, fc, x, prep.cache)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::FiniteDiffGradientPrep,
    ::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return fc(x), finite_difference_gradient!(grad, fc, x, prep.cache)
end

## Jacobian

struct FiniteDiffOneArgJacobianPrep{C} <: JacobianPrep
    cache::C
end

function DI.prepare_jacobian(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffOneArgJacobianPrep(cache)
end

function DI.jacobian(
    f, prep::FiniteDiffOneArgJacobianPrep, ::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_jacobian(fc, x, prep.cache)
end

function DI.value_and_jacobian(
    f, prep::FiniteDiffOneArgJacobianPrep, ::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    return y, finite_difference_jacobian(fc, x, prep.cache, y)
end

function DI.jacobian!(
    f,
    jac,
    prep::FiniteDiffOneArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return copyto!(jac, finite_difference_jacobian(fc, x, prep.cache; jac_prototype=jac))
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::FiniteDiffOneArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    return y,
    copyto!(jac, finite_difference_jacobian(fc, x, prep.cache, y; jac_prototype=jac))
end

## Hessian

struct FiniteDiffHessianPrep{C1,C2} <: HessianPrep
    gradient_cache::C1
    hessian_cache::C2
end

function DI.prepare_hessian(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    df = zero(y) .* x
    gradient_cache = GradientCache(df, x, fdtype(backend))
    hessian_cache = HessianCache(x, fdhtype(backend))
    return FiniteDiffHessianPrep(gradient_cache, hessian_cache)
end

function DI.hessian(
    f, prep::FiniteDiffHessianPrep, backend::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_hessian(fc, x, prep.hessian_cache)
end

function DI.hessian!(
    f,
    hess,
    prep::FiniteDiffHessianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    return finite_difference_hessian!(hess, fc, x, prep.hessian_cache)
end

function DI.value_gradient_and_hessian(
    f, prep::FiniteDiffHessianPrep, backend::AutoFiniteDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    grad = finite_difference_gradient(fc, x, prep.gradient_cache)
    hess = finite_difference_hessian(fc, x, prep.hessian_cache)
    return fc(x), grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::FiniteDiffHessianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    finite_difference_gradient!(grad, fc, x, prep.gradient_cache)
    finite_difference_hessian!(hess, fc, x, prep.hessian_cache)
    return fc(x), grad, hess
end
