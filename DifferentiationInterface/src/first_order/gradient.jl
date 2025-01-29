## Docstrings

"""
    prepare_gradient(f, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`gradient`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_gradient end

"""
    prepare!_gradient(f, prep, backend, x, [contexts...]) -> new_prep

Same behavior as [`prepare_gradient`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_gradient end

"""
    value_and_gradient(f, [prep,] backend, x, [contexts...]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`.

$(document_preparation("gradient"))
"""
function value_and_gradient end

"""
    value_and_gradient!(f, grad, [prep,] backend, x, [contexts...]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`, overwriting `grad`.

$(document_preparation("gradient"))
"""
function value_and_gradient! end

"""
    gradient(f, [prep,] backend, x, [contexts...]) -> grad

Compute the gradient of the function `f` at point `x`.

$(document_preparation("gradient"))
"""
function gradient end

"""
    gradient!(f, grad, [prep,] backend, x, [contexts...]) -> grad

Compute the gradient of the function `f` at point `x`, overwriting `grad`.

$(document_preparation("gradient"))
"""
function gradient! end

## Preparation

struct PullbackGradientPrep{E<:PullbackPrep} <: GradientPrep
    pullback_prep::E
end

function prepare_gradient(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    pullback_prep = prepare_pullback(f, backend, x, (true,), contexts...)
    return PullbackGradientPrep(pullback_prep)
end

## One argument

function value_and_gradient(
    f::F,
    prep::PullbackGradientPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, tx = value_and_pullback(f, prep.pullback_prep, backend, x, (true,), contexts...)
    return y, only(tx)
end

function value_and_gradient!(
    f::F,
    grad,
    prep::PullbackGradientPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, _ = value_and_pullback!(
        f, (grad,), prep.pullback_prep, backend, x, (true,), contexts...
    )
    return y, grad
end

function gradient(
    f::F,
    prep::PullbackGradientPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    tx = pullback(f, prep.pullback_prep, backend, x, (true,), contexts...)
    return only(tx)
end

function gradient!(
    f::F,
    grad,
    prep::PullbackGradientPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    pullback!(f, (grad,), prep.pullback_prep, backend, x, (true,), contexts...)
    return grad
end

## Shuffled

function shuffled_gradient(
    x, f::F, backend::AbstractADType, rewrap::Rewrap{C}, unannotated_contexts::Vararg{Any,C}
) where {F,C}
    return gradient(f, backend, x, rewrap(unannotated_contexts...)...)
end

function shuffled_gradient!(
    grad,
    x,
    f::F,
    backend::AbstractADType,
    rewrap::Rewrap{C},
    unannotated_contexts::Vararg{Any,C},
) where {F,C}
    return gradient!(f, grad, backend, x, rewrap(unannotated_contexts...)...)
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
    return gradient!(f, grad, prep, backend, x, rewrap(unannotated_contexts...)...)
end
