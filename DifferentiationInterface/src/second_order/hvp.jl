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

## Forward over forward

struct ForwardOverForwardHVPPrep{F,B,X,T,CC,E2<:PushforwardPrep} <: HVPPrep{F,B,X,T,CC}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    outer_pushforward_prep::E2
end

function ForwardOverForwardHVPPrep{F,B,X,T,CC}(
    outer_pushforward_prep::E
) where {F,B,X,T,CC,E}
    return ForwardOverForwardHVPPrep{F,B,X,T,CC,E}(outer_pushforward_prep)
end

function _prepare_hvp_aux(
    ::ForwardOverForward, f::F, backend::B, x::X, tx::T, contexts::Vararg{Context,C}
) where {F,B,X,T,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ForwardOverForwardHVPPrep{F,B,X,T,typeof(contexts)}(outer_pushforward_prep)
end

function hvp(
    f::F,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return value_and_pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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

## Forward over reverse

struct ForwardOverReverseHVPPrep{F,B,X,T,CC,E2<:PushforwardPrep} <: HVPPrep{F,B,X,T,CC}
    # pushforward of gradient
    outer_pushforward_prep::E2
end

function ForwardOverReverseHVPPrep{F,B,X,T,CC}(
    outer_pushforward_prep::E
) where {F,B,X,T,CC,E}
    return ForwardOverReverseHVPPrep{F,B,X,T,CC,E}(outer_pushforward_prep)
end

function _prepare_hvp_aux(
    ::ForwardOverReverse, f::F, backend::B, x::X, tx::T, contexts::Vararg{Context,C}
) where {F,B,X,T,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ForwardOverReverseHVPPrep{F,B,X,T,typeof(contexts)}(outer_pushforward_prep)
end

function hvp(
    f::F,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return value_and_pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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

struct ReverseOverForwardHVPPrep{F,B,X,T,CC,E2<:GradientPrep,E1<:GradientPrep} <:
       HVPPrep{F,B,X,T,CC}
    # gradient of pushforward
    outer_gradient_prep::E2
    gradient_prep::E1
end

function ReverseOverForwardHVPPrep{F,B,X,T,CC}(
    outer_gradient_prep::E2, gradient_prep::E1
) where {F,B,X,T,CC,E2,E1}
    return ReverseOverForwardHVPPrep{F,B,X,T,CC,E2,E1}(outer_gradient_prep, gradient_prep)
end

function _prepare_hvp_aux(
    ::ReverseOverForward, f::F, backend::B, x::X, tx::T, contexts::Vararg{Context,C}
) where {F,B,X,T,C}
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
    return ReverseOverForwardHVPPrep{F,B,X,T,typeof(contexts)}(
        outer_gradient_prep, gradient_prep
    )
end

function hvp(
    f::F,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
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
    check_prep(f, prep, backend, x, tx, contexts...)
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
    check_prep(f, prep, backend, x, tx, contexts...)
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
    check_prep(f, prep, backend, x, tx, contexts...)
    hvp!(f, tg, prep, backend, x, tx, contexts...)
    gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    return grad, tg
end

## Reverse over reverse

struct ReverseOverReverseHVPPrep{F,B,X,T,CC,E2<:PullbackPrep} <: HVPPrep{F,B,X,T,CC}
    # pullback of gradient
    outer_pullback_prep::E2
end

function ReverseOverReverseHVPPrep{F,B,X,T,CC}(
    outer_pullback_prep::E2
) where {F,B,X,T,CC,E2}
    return ReverseOverReverseHVPPrep{F,B,X,T,CC,E2}(outer_pullback_prep)
end

function _prepare_hvp_aux(
    ::ReverseOverReverse, f::F, backend::B, x::X, tx::T, contexts::Vararg{Context,C}
) where {F,B,X,T,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    outer_pullback_prep = prepare_pullback(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ReverseOverReverseHVPPrep{F,B,X,T,typeof(contexts)}(outer_pullback_prep)
end

function hvp(
    f::F,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
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
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    new_grad, _ = value_and_pullback!(
        shuffled_gradient, tg, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
    return copyto!(grad, new_grad), tg
end
