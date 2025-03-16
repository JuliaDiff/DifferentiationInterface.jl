## Docstrings

"""
    prepare_hvp(f, backend, x, tx, [contexts...]; strict=Val(false)) -> prep

$(docstring_prepare("hvp"))
"""
function prepare_hvp end

"""
    prepare!_hvp(f, backend, x, tx, [contexts...]) -> new_prep

$(docstring_prepare("hvp"))
"""
function prepare!_hvp end

"""
    prepare_hvp_same_point(f, backend, x, tx, [contexts...]; strict=Val(false)) -> prep_same

$(docstring_prepare("hvp"; samepoint=true))
"""
function prepare_hvp_same_point end

"""
    hvp(f, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(docstring_preparation_hint("hvp"; same_point=true))
"""
function hvp end

"""
    hvp!(f, tg, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `tg`.

$(docstring_preparation_hint("hvp"; same_point=true))
"""
function hvp! end

"""
    gradient_and_hvp(f, [prep,] backend, x, tx, [contexts...]) -> (grad, tg)

Compute the gradient and the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(docstring_preparation_hint("hvp"; same_point=true))
"""
function gradient_and_hvp end

"""
    gradient_and_hvp!(f, grad, tg, [prep,] backend, x, tx, [contexts...]) -> (grad, tg)

Compute the gradient and the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `grad` and `tg`.

$(docstring_preparation_hint("hvp"; same_point=true))
"""
function gradient_and_hvp! end

function prepare_hvp(
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
    strict::Val=Val(false),
) where {F,C}
    return _prepare_hvp_aux(
        hvp_mode(backend),
        inner_preparation_behavior(outer(backend)),
        f,
        backend,
        x,
        tx,
        contexts...;
        strict,
    )
end

## Forward over anything

struct ForwardOverAnythingHVPPrep{SIG,G,GO,GI,PO,PI} <: HVPPrep{SIG}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    _sig::Val{SIG}
    grad_buffer::G
    maybe_inner_gradient_prep::GO
    maybe_inner_gradient_in_prep::GI
    outer_pushforward_prep::PO
    outer_pushforward_in_prep::PI
end

function _prepare_hvp_aux(
    ::ForwardOverAnything,
    ::DontPrepareInner,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
    strict::Val,
) where {F,C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    grad_buffer = similar(x)
    rewrap = Rewrap(contexts...)
    # Outer pushforward
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...; strict
    )
    outer_pushforward_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pushforward(
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts...;
            strict,
        )
    else
        nothing
    end
    return ForwardOverAnythingHVPPrep(
        _sig, grad_buffer, (), (), outer_pushforward_prep, outer_pushforward_in_prep
    )
end

function _prepare_hvp_aux(
    ::ForwardOverAnything,
    ::PrepareInnerSimple,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
    strict::Val,
) where {F,C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    grad_buffer = similar(x)
    rewrap = Rewrap(contexts...)
    # Inner gradient
    inner_gradient_prep = prepare_gradient(f, inner(backend), x, contexts...; strict)
    inner_gradient_in_prep = inner_gradient_prep
    # Outer pushforward
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_contexts_in = (
        FunctionContext(f),
        PrepContext(inner_gradient_in_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...; strict
    )
    outer_pushforward_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pushforward(
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts_in...;
            strict,
        )
    else
        nothing
    end
    return ForwardOverAnythingHVPPrep(
        _sig,
        grad_buffer,
        (inner_gradient_prep,),
        (inner_gradient_in_prep,),
        outer_pushforward_prep,
        outer_pushforward_in_prep,
    )
end

function _prepare_hvp_aux(
    ::ForwardOverAnything,
    ::PrepareInnerOverload,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
    strict::Val,
) where {F,C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    grad_buffer = similar(x)
    rewrap = Rewrap(contexts...)
    # Inner gradient
    xo = overloaded_input(pushforward, shuffled_gradient, outer(backend), x, tx)
    xoi = overloaded_input(
        pushforward, shuffled_gradient!, grad_buffer, outer(backend), x, tx
    )
    contextso = adapt_eltype.(contexts, Ref(eltype(xo)))
    contextsoi = adapt_eltype.(contexts, Ref(eltype(xoi)))
    inner_gradient_prep = prepare_gradient(f, inner(backend), xo, contextso...; strict)
    inner_gradient_in_prep = prepare_gradient(f, inner(backend), xoi, contextsoi...; strict)
    # Outer pushforward
    new_contexts = (
        FunctionContext(f),
        PrepContext(inner_gradient_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_contexts_in = (
        FunctionContext(f),
        PrepContext(inner_gradient_in_prep),
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pushforward_prep = prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...; strict
    )
    outer_pushforward_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pushforward(
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts_in...;
            strict,
        )
    else
        nothing
    end
    return ForwardOverAnythingHVPPrep(
        _sig,
        grad_buffer,
        (inner_gradient_prep,),
        (inner_gradient_in_prep,),
        outer_pushforward_prep,
        outer_pushforward_in_prep,
    )
end

function hvp(
    f::F,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(PrepContext, maybe_inner_gradient_prep)...,
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
    check_prep(f, prep, backend, x, tx, contexts...)
    return _hvp_aux!(
        inplace_support(outer(backend)), f, tg, prep, backend, x, tx, contexts...
    )
end

function _hvp_aux!(
    ::InPlaceSupported,
    f::F,
    tg::NTuple,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; grad_buffer, maybe_inner_gradient_in_prep, outer_pushforward_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(PrepContext, maybe_inner_gradient_in_prep)...,
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return pushforward!(
        shuffled_gradient!,
        grad_buffer,
        tg,
        outer_pushforward_in_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
end

function _hvp_aux!(
    ::InPlaceNotSupported,
    f::F,
    tg::NTuple,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(PrepContext, maybe_inner_gradient_prep)...,
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
    check_prep(f, prep, backend, x, tx, contexts...)
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(PrepContext, maybe_inner_gradient_prep)...,
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
    check_prep(f, prep, backend, x, tx, contexts...)
    return _gradient_and_hvp_aux!(
        inplace_support(outer(backend)), f, grad, tg, prep, backend, x, tx, contexts...
    )
end

function _gradient_and_hvp_aux!(
    ::InPlaceSupported,
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; maybe_inner_gradient_in_prep, outer_pushforward_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(PrepContext, maybe_inner_gradient_in_prep)...,
        BackendContext(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    value_and_pushforward!(
        shuffled_gradient!,
        grad,
        tg,
        outer_pushforward_in_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return grad, tg
end

function _gradient_and_hvp_aux!(
    ::InPlaceNotSupported,
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardOverAnythingHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(PrepContext, maybe_inner_gradient_prep)...,
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

struct ReverseOverForwardHVPPrep{SIG,G2<:GradientPrep,G1<:GradientPrep} <: HVPPrep{SIG}
    # gradient of pushforward
    _sig::Val{SIG}
    outer_gradient_prep::G2
    gradient_prep::G1
end

function _prepare_hvp_aux(
    ::ReverseOverForward,
    ::InnerPreparationBehavior,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
    strict::Val,
) where {F,C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        BackendContext(inner(backend)),
        Constant(first(tx)),
        Constant(rewrap),
        contexts...,
    )
    outer_gradient_prep = prepare_gradient(
        shuffled_single_pushforward, outer(backend), x, new_contexts...; strict
    )
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...; strict)
    return ReverseOverForwardHVPPrep(_sig, outer_gradient_prep, gradient_prep)
end

function hvp(
    f::F,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
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

struct ReverseOverReverseHVPPrep{SIG,G,PO,PI} <: HVPPrep{SIG}
    # pullback of gradient
    _sig::Val{SIG}
    grad_buffer::G
    outer_pullback_prep::PO
    outer_pullback_in_prep::PI
end

function _prepare_hvp_aux(
    ::ReverseOverReverse,
    ::InnerPreparationBehavior,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C};
    strict::Val,
) where {F,C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    grad_buffer = similar(x)
    outer_pullback_prep = prepare_pullback(
        shuffled_gradient, outer(backend), x, tx, new_contexts...; strict
    )
    outer_pullback_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pullback(
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts...;
            strict,
        )
    else
        nothing
    end
    return ReverseOverReverseHVPPrep(
        _sig, grad_buffer, outer_pullback_prep, outer_pullback_in_prep
    )
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
    return _hvp_aux!(
        inplace_support(outer(backend)), f, tg, prep, backend, x, tx, contexts...
    )
end

function _hvp_aux!(
    ::InPlaceSupported,
    f::F,
    tg::NTuple,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; grad_buffer, outer_pullback_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    return pullback!(
        shuffled_gradient!,
        grad_buffer,
        tg,
        outer_pullback_in_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
end

function _hvp_aux!(
    ::InPlaceNotSupported,
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
    return _gradient_and_hvp_aux!(
        inplace_support(outer(backend)), f, grad, tg, prep, backend, x, tx, contexts...
    )
end

function _gradient_and_hvp_aux!(
    ::InPlaceSupported,
    f::F,
    grad,
    tg::NTuple,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; outer_pullback_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), BackendContext(inner(backend)), Constant(rewrap), contexts...
    )
    new_grad, _ = value_and_pullback!(
        shuffled_gradient!,
        grad,
        tg,
        outer_pullback_in_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return grad, tg
end

function _gradient_and_hvp_aux!(
    ::InPlaceNotSupported,
    f::F,
    grad,
    tg::NTuple,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
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
