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
    relstep = if isnothing(backend.relstep)
        default_relstep(fdtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep = isnothing(backend.absstep) ? relstep : backend.relstep
    ty = map(tx) do dx
        finite_difference_derivative(
            Base.Fix2(step, dx),
            zero(eltype(x)),
            fdtype(backend),
            eltype(y),
            y;
            relstep,
            absstep,
        )
    end
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, ty
end

## Derivative

struct FiniteDiffTwoArgDerivativePrep{C,R,A} <: DI.DerivativePrep
    cache::C
    relstep::R
    absstep::A
end

function DI.prepare_derivative(
    f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    df = similar(y)
    cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    relstep = if isnothing(backend.relstep)
        default_relstep(fdtype(backend), eltype(x))
    else
        backend.relstep
    end
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
    (; relstep, absstep) = prep
    der = finite_difference_gradient(fc!, x, prep.cache; relstep, absstep)
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
    (; relstep, absstep) = prep
    finite_difference_gradient!(der, fc!, x, prep.cache; relstep, absstep)
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
    (; relstep, absstep) = prep
    der = finite_difference_gradient(fc!, x, prep.cache; relstep, absstep)
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
    (; relstep, absstep) = prep
    finite_difference_gradient!(der, fc!, x, prep.cache; relstep, absstep)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianPrep{C,R,A} <: DI.JacobianPrep
    cache::C
    relstep::R
    absstep::A
end

function DI.prepare_jacobian(
    f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    relstep = if isnothing(backend.relstep)
        default_relstep(fdtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep = isnothing(backend.absstep) ? relstep : backend.relstep
    return FiniteDiffTwoArgJacobianPrep(cache, relstep, absstep)
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
    (; relstep, absstep) = prep
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep)
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
    (; relstep, absstep) = prep
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep)
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
    (; relstep, absstep) = prep
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep)
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
    (; relstep, absstep) = prep
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep)
    return jac
end
