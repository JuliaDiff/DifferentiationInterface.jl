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

## Preparation

struct ForwardOverForwardHVPPrep{E<:PushforwardPrep} <: HVPPrep
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    outer_pushforward_prep::E
end

struct ForwardOverReverseHVPPrep{E<:PushforwardPrep} <: HVPPrep
    # pushforward of gradient
    outer_pushforward_prep::E
end

struct ReverseOverForwardHVPPrep{E<:GradientPrep} <: HVPPrep
    # gradient of pushforward
    outer_gradient_prep::E
end

struct ReverseOverReverseHVPPrep{E<:PullbackPrep} <: HVPPrep
    # pullback of gradient
    outer_pullback_prep::E
end

function prepare_hvp(
    f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_hvp_aux(hvp_mode(backend), f, backend, x, tx, contexts...)
end

function _prepare_hvp_aux(
    ::ForwardOverForward,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ForwardOverForwardHVPPrep(outer_pushforward_prep)
end

function _prepare_hvp_aux(
    ::ForwardOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ForwardOverReverseHVPPrep(outer_pushforward_prep)
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
        Constant(f),
        Constant(inner(backend)),
        Constant(first(tx)),
        Constant(rewrap),
        contexts...,
    )
    outer_gradient_prep = prepare_gradient(
        shuffled_single_pushforward, outer(backend), x, new_contexts...
    )
    return ReverseOverForwardHVPPrep(outer_gradient_prep)
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
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    outer_pullback_prep = prepare_pullback(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ReverseOverReverseHVPPrep(outer_pullback_prep)
end

## One argument

function hvp(
    f::F,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    return pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function hvp(
    f::F,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    return pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
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
    (; outer_gradient_prep) = prep
    rewrap = Rewrap(contexts...)
    tg = map(tx) do dx
        gradient(
            shuffled_single_pushforward,
            outer_gradient_prep,
            outer(backend),
            x,
            Constant(f),
            Constant(inner(backend)),
            Constant(dx),
            Constant(rewrap),
            contexts...,
        )
    end
    return tg
end

function hvp(
    f::F,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    return pullback(
        shuffled_gradient, outer_pullback_prep, outer(backend), x, tx, new_contexts...
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
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
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

function hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
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
            Constant(f),
            Constant(inner(backend)),
            Constant(tx[b]),
            Constant(rewrap),
            contexts...,
        )
    end
    return tg
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
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (Constant(f), Constant(inner(backend)), Constant(rewrap), contexts...)
    return pullback!(
        shuffled_gradient, tg, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
end
