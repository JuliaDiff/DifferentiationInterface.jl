## Docstrings

"""
    prepare_gradient(f, backend, x, [contexts...]; strict=Val(false)) -> prep

$(docstring_prepare("gradient"))
"""
function prepare_gradient(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Val=Val(false)
) where {F,C}
    return prepare_gradient_nokwarg(strict, f, backend, x, contexts...)
end

"""
    prepare!_gradient(f, prep, backend, x, [contexts...]) -> new_prep

$(docstring_prepare!("gradient"))
"""
function prepare!_gradient(
    f::F, old_prep::GradientPrep, backend::AbstractADType, x, contexts::Vararg{Context,C};
) where {F,C}
    check_prep(f, old_prep, backend, x, contexts...)
    return prepare_gradient_nokwarg(is_strict(old_prep), f, backend, x, contexts...)
end

"""
    value_and_gradient(f, [prep,] backend, x, [contexts...]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`.

$(docstring_preparation_hint("gradient"))
"""
function value_and_gradient(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_gradient_nokwarg(Val(true), f, backend, x, contexts...)
    return value_and_gradient(f, prep, backend, x, contexts...)
end

"""
    value_and_gradient!(f, grad, [prep,] backend, x, [contexts...]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`, overwriting `grad`.

$(docstring_preparation_hint("gradient"))
"""
function value_and_gradient!(
    f::F, grad, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_gradient_nokwarg(Val(true), f, backend, x, contexts...)
    return value_and_gradient!(f, grad, prep, backend, x, contexts...)
end

"""
    gradient(f, [prep,] backend, x, [contexts...]) -> grad

Compute the gradient of the function `f` at point `x`.

$(docstring_preparation_hint("gradient"))
"""
function gradient(f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}) where {F,C}
    prep = prepare_gradient_nokwarg(Val(true), f, backend, x, contexts...)
    return gradient(f, prep, backend, x, contexts...)
end

"""
    gradient!(f, grad, [prep,] backend, x, [contexts...]) -> grad

Compute the gradient of the function `f` at point `x`, overwriting `grad`.

$(docstring_preparation_hint("gradient"))
"""
function gradient!(
    f::F, grad, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_gradient_nokwarg(Val(true), f, backend, x, contexts...)
    return gradient!(f, grad, prep, backend, x, contexts...)
end

## Preparation

struct PullbackGradientPrep{SIG,Y,E<:PullbackPrep} <: GradientPrep{SIG}
    _sig::Val{SIG}
    y::Y
    pullback_prep::E
end

function prepare_gradient_nokwarg(
    strict::Val, f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    _sig = signature(f, backend, x, contexts...; strict)
    y = f(x, map(unwrap, contexts)...)  # TODO: replace with output type inference?
    pullback_prep = prepare_pullback_nokwarg(
        strict, f, backend, x, (one(typeof(y)),), contexts...
    )
    return PullbackGradientPrep(_sig, y, pullback_prep)
end

## One argument

function value_and_gradient(
    f::F,
    prep::PullbackGradientPrep{SIG,Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,SIG,Y,C}
    check_prep(f, prep, backend, x, contexts...)
    y, tx = value_and_pullback(f, prep.pullback_prep, backend, x, (one(Y),), contexts...)
    return y, only(tx)
end

function value_and_gradient!(
    f::F,
    grad,
    prep::PullbackGradientPrep{SIG,Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,SIG,Y,C}
    check_prep(f, prep, backend, x, contexts...)
    y, _ = value_and_pullback!(
        f, (grad,), prep.pullback_prep, backend, x, (one(Y),), contexts...
    )
    return y, grad
end

function gradient(
    f::F,
    prep::PullbackGradientPrep{SIG,Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,SIG,Y,C}
    check_prep(f, prep, backend, x, contexts...)
    tx = pullback(f, prep.pullback_prep, backend, x, (one(Y),), contexts...)
    return only(tx)
end

function gradient!(
    f::F,
    grad,
    prep::PullbackGradientPrep{SIG,Y},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,SIG,Y,C}
    check_prep(f, prep, backend, x, contexts...)
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
    backend::AbstractADType,
    rewrap::Rewrap{C},
    unannotated_contexts::Vararg{Any,C},
) where {F,C}
    gradient!(f, grad, backend, x, rewrap(unannotated_contexts...)...)
    return nothing
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
