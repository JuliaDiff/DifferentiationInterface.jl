## Pushforward

function DI.prepare_pushforward(
    f!, y, ::AutoFiniteDiff, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoPushforwardPrep()
end

function DI.value_and_pushforward(
    f!,
    y,
    ::DI.NoPushforwardPrep,
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    function step(t::Number, dx)
        new_y = similar(y)
        f!(new_y, x .+ t .* dx, map(DI.unwrap, contexts)...)
        return new_y
    end
    ty = map(tx) do dx
        finite_difference_derivative(
            Base.Fix2(step, dx), zero(eltype(x)), fdtype(backend), eltype(y), y
        )
    end
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, ty
end

## Derivative

struct FiniteDiffTwoArgDerivativePrep{C} <: DI.DerivativePrep
    cache::C
    relstep
    absstep
end

function DI.prepare_derivative(
    f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    df = similar(y)
    cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    relstep =
        isnothing(backend.relstep) ? default_relstep(fdtype, eltype(x)) : backend.relstep
    absstep = isnothing(backend.absstep) ? relstep : backend.relstep
    return FiniteDiffTwoArgDerivativePrep(cache, relstep, absstep)
end

function DI.value_and_derivative(
    f!,
    y,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    fc!(y, x)
    der = finite_difference_gradient(fc!, x, prep.cache)
    return y, der
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    fc!(y, x)
    finite_difference_gradient!(der, fc!, x, prep.cache)
    return y, der
end

function DI.derivative(
    f!,
    y,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    fc!(y, x)
    der = finite_difference_gradient(fc!, x, prep.cache)
    return der
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    finite_difference_gradient!(der, fc!, x, prep.cache)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianPrep{C} <: DI.JacobianPrep
    cache::C
end

function DI.prepare_jacobian(
    f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffTwoArgJacobianPrep(cache)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(
        jac, fc!, x, prep.cache; relstep=prep.relstep, absstep=prep.absstep
    )
    fc!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    finite_difference_jacobian!(
        jac, fc!, x, prep.cache; relstep=prep.relstep, absstep=prep.absstep
    )
    fc!(y, x)
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(
        jac, fc!, x, prep.cache; relstep=prep.relstep, absstep=prep.absstep
    )
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    finite_difference_jacobian!(
        jac, fc!, x, prep.cache; relstep=prep.relstep, absstep=prep.absstep
    )
    return jac
end
