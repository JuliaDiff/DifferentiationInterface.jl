## Docstrings

"""
    prepare_derivative(f,     backend, x, [contexts...]) -> prep
    prepare_derivative(f!, y, backend, x, [contexts...]) -> prep

$(docstring_prepare("derivative"; inplace=true))
"""
function prepare_derivative end

"""
    prepare!_derivative(f,     prep, backend, x, [contexts...]) -> new_prep
    prepare!_derivative(f!, y, prep, backend, x, [contexts...]) -> new_prep

$(docstring_prepare!("derivative"))
"""
function prepare!_derivative end

"""
    value_and_derivative(f,     [prep,] backend, x, [contexts...]) -> (y, der)
    value_and_derivative(f!, y, [prep,] backend, x, [contexts...]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`.

$(docstring_preparation_hint("derivative"))
"""
function value_and_derivative end

"""
    value_and_derivative!(f,     der, [prep,] backend, x, [contexts...]) -> (y, der)
    value_and_derivative!(f!, y, der, [prep,] backend, x, [contexts...]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`, overwriting `der`.

$(docstring_preparation_hint("derivative"))
"""
function value_and_derivative! end

"""
    derivative(f,     [prep,] backend, x, [contexts...]) -> der
    derivative(f!, y, [prep,] backend, x, [contexts...]) -> der

Compute the derivative of the function `f` at point `x`.

$(docstring_preparation_hint("derivative"))
"""
function derivative end

"""
    derivative!(f,     der, [prep,] backend, x, [contexts...]) -> der
    derivative!(f!, y, der, [prep,] backend, x, [contexts...]) -> der

Compute the derivative of the function `f` at point `x`, overwriting `der`.

$(docstring_preparation_hint("derivative"))
"""
function derivative! end

## Preparation

struct PushforwardDerivativePrep{E<:PushforwardPrep} <: DerivativePrep
    pushforward_prep::E
end

function prepare_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    pushforward_prep = prepare_pushforward(f, backend, x, (one(x),), contexts...)
    return PushforwardDerivativePrep(pushforward_prep)
end

function prepare_derivative(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    pushforward_prep = prepare_pushforward(f!, y, backend, x, (one(x),), contexts...)
    return PushforwardDerivativePrep(pushforward_prep)
end

## One argument

function value_and_derivative(
    f::F,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, ty = value_and_pushforward(
        f, prep.pushforward_prep, backend, x, (one(x),), contexts...
    )
    return y, only(ty)
end

function value_and_derivative!(
    f::F,
    der,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, _ = value_and_pushforward!(
        f, (der,), prep.pushforward_prep, backend, x, (one(x),), contexts...
    )
    return y, der
end

function derivative(
    f::F,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    ty = pushforward(f, prep.pushforward_prep, backend, x, (one(x),), contexts...)
    return only(ty)
end

function derivative!(
    f::F,
    der,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    pushforward!(f, (der,), prep.pushforward_prep, backend, x, (one(x),), contexts...)
    return der
end

## Two arguments

function value_and_derivative(
    f!::F,
    y,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, ty = value_and_pushforward(
        f!, y, prep.pushforward_prep, backend, x, (one(x),), contexts...
    )
    return y, only(ty)
end

function value_and_derivative!(
    f!::F,
    y,
    der,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, _ = value_and_pushforward!(
        f!, y, (der,), prep.pushforward_prep, backend, x, (one(x),), contexts...
    )
    return y, der
end

function derivative(
    f!::F,
    y,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    ty = pushforward(f!, y, prep.pushforward_prep, backend, x, (one(x),), contexts...)
    return only(ty)
end

function derivative!(
    f!::F,
    y,
    der,
    prep::PushforwardDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    pushforward!(f!, y, (der,), prep.pushforward_prep, backend, x, (one(x),), contexts...)
    return der
end

## Shuffled

function shuffled_derivative(
    x, f::F, backend::AbstractADType, rewrap::Rewrap{C}, unannotated_contexts::Vararg{Any,C}
) where {F,C}
    return derivative(f, backend, x, rewrap(unannotated_contexts...)...)
end
