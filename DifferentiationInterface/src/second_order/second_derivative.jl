## Docstrings

"""
    prepare_second_derivative(f, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`second_derivative`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_second_derivative end

"""
    second_derivative(f, [prep,] backend, x, [contexts...]) -> der2

Compute the second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function second_derivative end

"""
    second_derivative!(f, der2, [prep,] backend, x, [contexts...]) -> der2

Compute the second derivative of the function `f` at point `x`, overwriting `der2`.

$(document_preparation("second_derivative"))
"""
function second_derivative! end

"""
    value_derivative_and_second_derivative(f, [prep,] backend, x, [contexts...]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative end

"""
    value_derivative_and_second_derivative!(f, der, der2, [prep,] backend, x, [contexts...]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`, overwriting `der` and `der2`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative! end

## Preparation

struct DerivativeSecondDerivativePrep{E<:DerivativePrep} <: SecondDerivativePrep
    outer_derivative_prep::E
end

function prepare_second_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    outer_derivative_prep = prepare_derivative(
        shuffled_derivative, outer(backend), x, new_contexts...
    )
    return DerivativeSecondDerivativePrep(outer_derivative_prep)
end

## One argument

function second_derivative(
    f::F,
    prep::DerivativeSecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_derivative_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return derivative(
        shuffled_derivative, outer_derivative_prep, outer(backend), x, new_contexts...
    )
end

function value_derivative_and_second_derivative(
    f::F,
    prep::DerivativeSecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_derivative_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    y = f(x, map(unwrap, contexts)...)
    der, der2 = value_and_derivative(
        shuffled_derivative, outer_derivative_prep, outer(backend), x, new_contexts...
    )
    return y, der, der2
end

function second_derivative!(
    f::F,
    der2,
    prep::SecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_derivative_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return derivative!(
        shuffled_derivative, der2, outer_derivative_prep, outer(backend), x, new_contexts...
    )
end

function value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    prep::SecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_derivative_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    y = f(x, map(unwrap, contexts)...)
    new_der, _ = value_and_derivative!(
        shuffled_derivative, der2, outer_derivative_prep, outer(backend), x, new_contexts...
    )
    return y, copyto!(der, new_der), der2
end
