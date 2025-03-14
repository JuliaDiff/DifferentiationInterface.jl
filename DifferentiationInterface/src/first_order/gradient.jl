## Docstrings

"""
    prepare_gradient(f, backend, x, [contexts...]) -> prep

$(docstring_prepare("gradient"))
"""
function prepare_gradient end

"""
    prepare!_gradient(f, prep, backend, x, [contexts...]) -> new_prep

$(docstring_prepare!("gradient"))
"""
function prepare!_gradient end

"""
    value_and_gradient(f, [prep,] backend, x, [contexts...]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`.

$(docstring_preparation_hint("gradient"))
"""
function value_and_gradient end

"""
    value_and_gradient!(f, grad, [prep,] backend, x, [contexts...]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`, overwriting `grad`.

$(docstring_preparation_hint("gradient"))
"""
function value_and_gradient! end

"""
    gradient(f, [prep,] backend, x, [contexts...]) -> grad

Compute the gradient of the function `f` at point `x`.

$(docstring_preparation_hint("gradient"))
"""
function gradient end

"""
    gradient!(f, grad, [prep,] backend, x, [contexts...]) -> grad

Compute the gradient of the function `f` at point `x`, overwriting `grad`.

$(docstring_preparation_hint("gradient"))
"""
function gradient! end

## Preparation

struct PullbackGradientPrep{Y,E<:PullbackPrep} <: GradientPrep
    pullback_prep::E
end

function prepare_gradient(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    y = f(x, map(unwrap, contexts)...)  # TODO: replace with output type inference?
    pullback_prep = prepare_pullback(f, backend, x, (true,), contexts...)
    return PullbackGradientPrep{typeof(y),typeof(pullback_prep)}(pullback_prep)
end

## One argument

function value_and_gradient(
    f::F,
    prep::PullbackGradientPrep{Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,Y,C}
    y, tx = value_and_pullback(f, prep.pullback_prep, backend, x, (one(Y),), contexts...)
    return y, only(tx)
end

function value_and_gradient!(
    f::F,
    grad,
    prep::PullbackGradientPrep{Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,Y,C}
    y, _ = value_and_pullback!(
        f, (grad,), prep.pullback_prep, backend, x, (one(Y),), contexts...
    )
    return y, grad
end

function gradient(
    f::F,
    prep::PullbackGradientPrep{Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,Y,C}
    tx = pullback(f, prep.pullback_prep, backend, x, (one(Y),), contexts...)
    return only(tx)
end

function gradient!(
    f::F,
    grad,
    prep::PullbackGradientPrep{Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,Y,C}
    pullback!(f, (grad,), prep.pullback_prep, backend, x, (one(Y),), contexts...)
    return grad
end

## Shuffled

function shuffled_gradient(
    x, f::F, backend::AbstractADType, rewrap::Rewrap{C}, unannotated_contexts::Vararg{Any,C}
) where {F,C}
    return gradient(f, backend, x, rewrap(unannotated_contexts...)...)
end

function shuffled_gradient(
    x,
    f::F,
    prep::GradientPrep,
    backend::AbstractADType,
    rewrap::Rewrap{C},
    unannotated_contexts::Vararg{Any,C},
) where {F,C}
    return gradient(f, prep, backend, x, rewrap(unannotated_contexts...)...)
end

function shuffled_gradient!(
    grad,
    x,
    f::F,
    prep::GradientPrep,
    backend::AbstractADType,
    rewrap::Rewrap{C},
    unannotated_contexts::Vararg{Any,C},
) where {F,C}
    gradient!(f, grad, prep, backend, x, rewrap(unannotated_contexts...)...)
    return nothing
end
