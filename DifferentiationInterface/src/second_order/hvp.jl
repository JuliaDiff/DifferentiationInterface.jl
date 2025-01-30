## Docstrings

"""
    prepare_hvp(f, backend, x, tx, [contexts...]) -> prep

Create a `prep` object that can be given to [`hvp`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp end

"""
    prepare!_hvp(f, backend, x, tx, [contexts...]) -> new_prep

Same behavior as [`prepare_hvp`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_hvp end

"""
    prepare_hvp_same_point(f, backend, x, tx, [contexts...]) -> prep_same

Create an `prep_same` object that can be given to [`hvp`](@ref) and its variants _if they are applied at the same point `x` and with the same `contexts`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp_same_point end

"""
    hvp(f, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp end

"""
    hvp!(f, tg, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `tg`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp! end

"""
    gradient_and_hvp(f, [prep,] backend, x, tx, [contexts...]) -> (grad, tg)

Compute the gradient and the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(document_preparation("hvp"; same_point=true))
"""
function gradient_and_hvp end

"""
    gradient_and_hvp!(f, grad, tg, [prep,] backend, x, tx, [contexts...]) -> (grad, tg)

Compute the gradient and the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `grad` and `tg`.

$(document_preparation("hvp"; same_point=true))
"""
function gradient_and_hvp! end

function prepare_hvp(
    f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_hvp_aux(hvp_mode(backend), f, backend, x, tx, contexts...)
end

## Forward over anything

struct ForwardOverAnythingHVPPrep{E1<:GradientPrep,E2<:PushforwardPrep} <: HVPPrep
    inner_gradient_prep::E1
    outer_pushforward_prep::E2
end

function _prepare_hvp_aux(
    ::Union{ForwardOverForward,ForwardOverReverse},
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    new_contexts_unknown_prep = (
        FunctionContext(f),
        UnknownContext(),  # placeholder for inner_gradient_prep
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    xo = overloaded_input_example(
        pushforward, shuffled_gradient, outer(backend), x, tx, new_contexts_unknown_prep...
    )
    copyto!(xo, x)
    inner_gradient_prep = prepare_gradient(f, inner(backend), xo, contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ForwardOverAnythingHVPPrep(inner_gradient_prep, outer_pushforward_prep)
end

function hvp(
    f::F,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return pushforward!(
        shuffled_gradient,
        tg,
        outer_pushforward_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
end

function gradient_and_hvp(
    f::F,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return value_and_pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_grad, _ = value_and_pushforward!(
        shuffled_gradient,
        tg,
        outer_pushforward_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return copyto!(grad, new_grad), tg
end

## Reverse over forward

struct ReverseOverForwardHVPPrep{E2<:GradientPrep,E1<:GradientPrep} <: HVPPrep
    outer_gradient_prep::E2
    gradient_prep::E1
end

function _prepare_hvp_aux(
    ::ReverseOverForward,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        BackendContext(inner(backend)),
        Constant(first(tx)),
        Constant(rewrap),
        contexts...,
    )
    outer_gradient_prep = prepare_gradient(
        shuffled_single_pushforward, outer(backend), x, new_contexts...
    )
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    return ReverseOverForwardHVPPrep(outer_gradient_prep, gradient_prep)
end

function hvp(
    f::F,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_gradient_prep) = prep
    rewrap = Rewrap(contexts...)
    tg = map(tx) do dx
        gradient(
            shuffled_single_pushforward,
            outer_gradient_prep,
            outer(backend),
            x,
            FunctionContext(f),
            BackendContext(inner(backend)),
            Constant(dx),
            Constant(rewrap),
            contexts...,
        )
    end
    return tg
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_gradient_prep) = prep
    rewrap = Rewrap(contexts...)
    for b in eachindex(tx, tg)
        gradient!(
            shuffled_single_pushforward,
            tg[b],
            outer_gradient_prep,
            outer(backend),
            x,
            FunctionContext(f),
            BackendContext(inner(backend)),
            Constant(tx[b]),
            Constant(rewrap),
            contexts...,
        )
    end
    return tg
end

function gradient_and_hvp(
    f::F,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    tg = hvp(f, prep, backend, x, tx, contexts...)
    grad = gradient(f, prep.gradient_prep, inner(backend), x, contexts...)
    return grad, tg
end

function gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    hvp!(f, tg, prep, backend, x, tx, contexts...)
    gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    return grad, tg
end

## Reverse over reverse

struct ReverseOverReverseHVPPrep{E1<:GradientPrep,E2<:PullbackPrep} <: HVPPrep
    inner_gradient_prep::E1
    outer_pullback_prep::E2
end

function _prepare_hvp_aux(
    ::ReverseOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    new_contexts_unknown_prep = (
        FunctionContext(f),
        UnknownContext(),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    xo = overloaded_input_example(
        pullback, shuffled_gradient, backend, x, tx, new_contexts_unknown_prep...
    )
    copyto!(xo, x)
    inner_gradient_prep = prepare_gradient(f, backend, xo, contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pullback_prep = prepare_pullback(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ReverseOverReverseHVPPrep(inner_gradient_prep, outer_pullback_prep)
end

function hvp(
    f::F,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return pullback(
        shuffled_gradient, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return pullback!(
        shuffled_gradient, tg, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
end

function gradient_and_hvp(
    f::F,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return value_and_pullback(
        shuffled_gradient, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
end

function gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_grad, _ = value_and_pullback!(
        shuffled_gradient, tg, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
    return copyto!(grad, new_grad), tg
end
