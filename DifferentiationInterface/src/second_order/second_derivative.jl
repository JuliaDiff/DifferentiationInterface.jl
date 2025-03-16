## Docstrings

"""
    prepare_second_derivative(f, backend, x, [contexts...]; strict=Val(false)) -> prep

$(docstring_prepare("second_derivative"))
"""
function prepare_second_derivative end

"""
    prepare!_second_derivative(f, prep, backend, x, [contexts...]) -> new_prep

$(docstring_prepare!("second_derivative"))
"""
function prepare!_second_derivative end

"""
    second_derivative(f, [prep,] backend, x, [contexts...]) -> der2

Compute the second derivative of the function `f` at point `x`.

$(docstring_preparation_hint("second_derivative"))
"""
function second_derivative end

"""
    second_derivative!(f, der2, [prep,] backend, x, [contexts...]) -> der2

Compute the second derivative of the function `f` at point `x`, overwriting `der2`.

$(docstring_preparation_hint("second_derivative"))
"""
function second_derivative! end

"""
    value_derivative_and_second_derivative(f, [prep,] backend, x, [contexts...]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`.

$(docstring_preparation_hint("second_derivative"))
"""
function value_derivative_and_second_derivative end

"""
    value_derivative_and_second_derivative!(f, der, der2, [prep,] backend, x, [contexts...]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`, overwriting `der` and `der2`.

$(docstring_preparation_hint("second_derivative"))
"""
function value_derivative_and_second_derivative! end

## Preparation

struct DerivativeSecondDerivativePrep{SIG,E<:DerivativePrep} <: SecondDerivativePrep{SIG}
    _sig::Val{SIG}
    outer_derivative_prep::E
end

function prepare_second_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Val=Val(false)
) where {F,C}
    _sig = signature(f, backend, x, contexts...; strict)
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    outer_derivative_prep = prepare_derivative(
        shuffled_derivative, outer(backend), x, new_contexts...; strict
    )
    return DerivativeSecondDerivativePrep(_sig, outer_derivative_prep)
end

## One argument

function second_derivative(
    f::F,
    prep::DerivativeSecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
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
    check_prep(f, prep, backend, x, contexts...)
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
    check_prep(f, prep, backend, x, contexts...)
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
    check_prep(f, prep, backend, x, contexts...)
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
