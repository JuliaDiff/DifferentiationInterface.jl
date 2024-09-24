## Docstrings

"""
    prepare_derivative(f,     backend, x, [contexts...]) -> prep
    prepare_derivative(f!, y, backend, x, [contexts...]) -> prep

Create an `prep` object that can be given to [`derivative`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_derivative end

"""
    value_and_derivative(f,     [prep,] backend, x, [contexts...]) -> (y, der)
    value_and_derivative(f!, y, [prep,] backend, x, [contexts...]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`.

$(document_preparation("derivative"))
"""
function value_and_derivative end

"""
    value_and_derivative!(f,     der, [prep,] backend, x, [contexts...]) -> (y, der)
    value_and_derivative!(f!, y, der, [prep,] backend, x, [contexts...]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`, overwriting `der`.

$(document_preparation("derivative"))
"""
function value_and_derivative! end

"""
    derivative(f,     [prep,] backend, x, [contexts...]) -> der
    derivative(f!, y, [prep,] backend, x, [contexts...]) -> der

Compute the derivative of the function `f` at point `x`.

$(document_preparation("derivative"))
"""
function derivative end

"""
    derivative!(f,     der, [prep,] backend, x, [contexts...]) -> der
    derivative!(f!, y, der, [prep,] backend, x, [contexts...]) -> der

Compute the derivative of the function `f` at point `x`, overwriting `der`.

$(document_preparation("derivative"))
"""
function derivative! end

## Preparation

struct PushforwardDerivativePrep{E<:PushforwardPrep} <: DerivativePrep
    pushforward_prep::E
end

function prepare_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    pushforward_prep = prepare_pushforward(f, backend, x, Tangents(one(x)), contexts...)
    return PushforwardDerivativePrep(pushforward_prep)
end

function prepare_derivative(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    pushforward_prep = prepare_pushforward(f!, y, backend, x, Tangents(one(x)), contexts...)
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
        f, prep.pushforward_prep, backend, x, Tangents(one(x)), contexts...
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
        f, Tangents(der), prep.pushforward_prep, backend, x, Tangents(one(x)), contexts...
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
    ty = pushforward(f, prep.pushforward_prep, backend, x, Tangents(one(x)), contexts...)
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
    pushforward!(
        f, Tangents(der), prep.pushforward_prep, backend, x, Tangents(one(x)), contexts...
    )
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
        f!, y, prep.pushforward_prep, backend, x, Tangents(one(x)), contexts...
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
        f!,
        y,
        Tangents(der),
        prep.pushforward_prep,
        backend,
        x,
        Tangents(one(x)),
        contexts...,
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
    ty = pushforward(
        f!, y, prep.pushforward_prep, backend, x, Tangents(one(x)), contexts...
    )
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
    pushforward!(
        f!,
        y,
        Tangents(der),
        prep.pushforward_prep,
        backend,
        x,
        Tangents(one(x)),
        contexts...,
    )
    return der
end
