## Docstrings

"""
    prepare_hvp(f, backend, x, tx, [contexts...]; strict=Val(true)) -> prep

$(docstring_prepare("hvp"))
"""
function prepare_hvp(
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_hvp_nokwarg(strict, f, backend, x, tx, contexts...)
end

"""
    prepare!_hvp(f, backend, x, tx, [contexts...]) -> new_prep

$(docstring_prepare("hvp"))
"""
function prepare!_hvp(
        f::F,
        old_prep::HVPPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, old_prep, backend, x, tx, contexts...)
    return prepare_hvp_nokwarg(is_strict(old_prep), f, backend, x, tx, contexts...)
end

"""
    prepare_hvp_same_point(f, backend, x, tx, [contexts...]; strict=Val(true)) -> prep_same

$(docstring_prepare("hvp"; samepoint = true))
"""
function prepare_hvp_same_point(
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_hvp_same_point_nokwarg(strict, f, backend, x, tx, contexts...)
end

function prepare_hvp_same_point_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hvp_nokwarg(strict, f, backend, x, tx, contexts...)
    return prepare_hvp_same_point(f, prep, backend, x, tx, contexts...)
end

function prepare_hvp_same_point(
        f::F, prep::HVPPrep, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    return prep
end

"""
    hvp(f, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(docstring_preparation_hint("hvp"; same_point = true))
"""
function hvp(f::F, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}) where {F, C}
    prep = prepare_hvp_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return hvp(f, prep, backend, x, tx, contexts...)
end

"""
    hvp!(f, tg, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `tg`.

$(docstring_preparation_hint("hvp"; same_point = true))
"""
function hvp!(
        f::F, tg, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hvp_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return hvp!(f, tg, prep, backend, x, tx, contexts...)
end

"""
    gradient_and_hvp(f, [prep,] backend, x, tx, [contexts...]) -> (grad, tg)

Compute the gradient and the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(docstring_preparation_hint("hvp"; same_point = true))
"""
function gradient_and_hvp(
        f::F, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hvp_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return gradient_and_hvp(f, prep, backend, x, tx, contexts...)
end

"""
    gradient_and_hvp!(f, grad, tg, [prep,] backend, x, tx, [contexts...]) -> (grad, tg)

Compute the gradient and the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `grad` and `tg`.

$(docstring_preparation_hint("hvp"; same_point = true))
"""
function gradient_and_hvp!(
        f::F, grad, tg, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hvp_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return gradient_and_hvp!(f, grad, tg, prep, backend, x, tx, contexts...)
end

## Preparation

function prepare_hvp_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    return _prepare_hvp_aux(
        strict,
        hvp_mode(backend),
        inner_preparation_behavior(outer(backend)),
        f,
        backend,
        x,
        tx,
        contexts...
    )
end

## Forward over anything

struct ForwardOverAnythingHVPPrep{SIG, G, GO, GI, PO, PI} <: HVPPrep{SIG}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    _sig::Val{SIG}
    grad_buffer::G
    maybe_inner_gradient_prep::GO
    maybe_inner_gradient_in_prep::GI
    outer_pushforward_prep::PO
    outer_pushforward_in_prep::PI
end

function _prepare_hvp_aux(
        strict::Val,
        ::ForwardOverAnything,
        ::DontPrepareInner,
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    grad_buffer = similar(x)
    rewrap = Rewrap(contexts...)
    # Outer pushforward
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
    )
    outer_pushforward_prep = prepare_pushforward_nokwarg(
        strict, shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    outer_pushforward_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pushforward_nokwarg(
            strict,
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts...
        )
    else
        nothing
    end
    return ForwardOverAnythingHVPPrep(
        _sig, grad_buffer, (), (), outer_pushforward_prep, outer_pushforward_in_prep
    )
end

function _prepare_hvp_aux(
        strict::Val,
        ::ForwardOverAnything,
        ::PrepareInnerSimple,
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    grad_buffer = similar(x)
    rewrap = Rewrap(contexts...)
    # Inner gradient
    inner_gradient_prep = prepare_gradient_nokwarg(
        strict, f, inner(backend), x, contexts...
    )
    inner_gradient_in_prep = inner_gradient_prep
    # Outer pushforward
    new_contexts = (
        FunctionContext(f),
        ConstantOrCache(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_contexts_in = (
        FunctionContext(f),
        ConstantOrCache(inner_gradient_in_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pushforward_prep = prepare_pushforward_nokwarg(
        strict, shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    outer_pushforward_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pushforward_nokwarg(
            strict,
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts_in...
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
        strict::Val,
        ::ForwardOverAnything,
        ::PrepareInnerOverload,
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
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
    inner_gradient_prep = prepare_gradient_nokwarg(
        strict, f, inner(backend), xo, contextso...
    )
    inner_gradient_in_prep = prepare_gradient_nokwarg(
        strict, f, inner(backend), xoi, contextsoi...
    )
    # Outer pushforward
    new_contexts = (
        FunctionContext(f),
        ConstantOrCache(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_contexts_in = (
        FunctionContext(f),
        ConstantOrCache(inner_gradient_in_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pushforward_prep = prepare_pushforward_nokwarg(
        strict, shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    outer_pushforward_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pushforward_nokwarg(
            strict,
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts_in...
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(ConstantOrCache, maybe_inner_gradient_prep)...,
        Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; grad_buffer, maybe_inner_gradient_in_prep, outer_pushforward_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(ConstantOrCache, maybe_inner_gradient_in_prep)...,
        Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(ConstantOrCache, maybe_inner_gradient_prep)...,
        Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(ConstantOrCache, maybe_inner_gradient_prep)...,
        Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; maybe_inner_gradient_in_prep, outer_pushforward_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(ConstantOrCache, maybe_inner_gradient_in_prep)...,
        Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; maybe_inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        map(ConstantOrCache, maybe_inner_gradient_prep)...,
        Constant(inner(backend)),
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
    return copy!(grad, new_grad), tg
end

## Reverse over forward

struct ReverseOverForwardHVPPrep{SIG, G2 <: GradientPrep, G1 <: GradientPrep} <: HVPPrep{SIG}
    # gradient of pushforward
    _sig::Val{SIG}
    outer_gradient_prep::G2
    gradient_prep::G1
end

function _prepare_hvp_aux(
        strict::Val,
        ::ReverseOverForward,
        ::InnerPreparationBehavior,
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f),
        Constant(inner(backend)),
        Constant(first(tx)),
        Constant(rewrap),
        contexts...,
    )
    outer_gradient_prep = prepare_gradient_nokwarg(
        strict, shuffled_single_pushforward, outer(backend), x, new_contexts...
    )
    gradient_prep = prepare_gradient_nokwarg(strict, f, inner(backend), x, contexts...)
    return ReverseOverForwardHVPPrep(_sig, outer_gradient_prep, gradient_prep)
end

function hvp(
        f::F,
        prep::ReverseOverForwardHVPPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
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
            Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
            Constant(inner(backend)),
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    hvp!(f, tg, prep, backend, x, tx, contexts...)
    gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    return grad, tg
end

## Reverse over reverse

struct ReverseOverReverseHVPPrep{SIG, G, PO, PI} <: HVPPrep{SIG}
    # pullback of gradient
    _sig::Val{SIG}
    grad_buffer::G
    outer_pullback_prep::PO
    outer_pullback_in_prep::PI
end

function _prepare_hvp_aux(
        strict::Val,
        ::ReverseOverReverse,
        ::InnerPreparationBehavior,
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
    )
    grad_buffer = similar(x)
    outer_pullback_prep = prepare_pullback_nokwarg(
        strict, shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    outer_pullback_in_prep = if inplace_support(outer(backend)) isa InPlaceSupported
        prepare_pullback_nokwarg(
            strict,
            shuffled_gradient!,
            grad_buffer,
            outer(backend),
            x,
            tx,
            new_contexts...
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; grad_buffer, outer_pullback_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; outer_pullback_in_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    (; outer_pullback_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        FunctionContext(f), Constant(inner(backend)), Constant(rewrap), contexts...,
    )
    new_grad, _ = value_and_pullback!(
        shuffled_gradient, tg, outer_pullback_prep, outer(backend), x, tx, new_contexts...
    )
    return copy!(grad, new_grad), tg
end
