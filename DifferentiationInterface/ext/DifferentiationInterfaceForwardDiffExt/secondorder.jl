struct ForwardDiffOverSomethingHVPPrep{
    G,E1<:DI.GradientPrep,E2<:DI.PushforwardPrep,E2IP<:DI.PushforwardPrep
} <: DI.HVPPrep
    grad_buffer::G
    inner_gradient_prep::E1
    outer_pushforward_prep::E2
    outer_pushforward_prep_inplace::E2IP
end

function DI.prepare_hvp(
    f::F,
    backend::DI.SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    T = tag_type(DI.shuffled_gradient, DI.outer(backend), x)
    xdual = make_dual(T, x, tx)
    inner_gradient_prep = DI.prepare_gradient(f, DI.inner(backend), xdual, contexts...)
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    grad_buffer = similar(x)
    outer_pushforward_prep = DI.prepare_pushforward(
        DI.shuffled_gradient, DI.outer(backend), x, tx, new_contexts...
    )
    outer_pushforward_prep_inplace = DI.prepare_pushforward(
        DI.shuffled_gradient!, grad_buffer, DI.outer(backend), x, tx, new_contexts...
    )
    return ForwardDiffOverSomethingHVPPrep(
        grad_buffer,
        inner_gradient_prep,
        outer_pushforward_prep,
        outer_pushforward_prep_inplace,
    )
end

function DI.hvp(
    f::F,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::DI.SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    return DI.pushforward(
        DI.shuffled_gradient,
        outer_pushforward_prep,
        DI.outer(backend),
        x,
        tx,
        new_contexts...,
    )
end

function DI.hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::DI.SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    (; grad_buffer, inner_gradient_prep, outer_pushforward_prep_inplace) = prep
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    return DI.pushforward!(
        DI.shuffled_gradient!,
        grad_buffer,
        tg,
        outer_pushforward_prep_inplace,
        DI.outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return tg
end

function DI.gradient_and_hvp(
    f::F,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::DI.SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    return DI.value_and_pushforward(
        DI.shuffled_gradient,
        outer_pushforward_prep,
        DI.outer(backend),
        x,
        tx,
        new_contexts...,
    )
end

function DI.gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::DI.SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep_inplace) = prep
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    return DI.value_and_pushforward!(
        DI.shuffled_gradient!,
        grad,
        tg,
        outer_pushforward_prep_inplace,
        DI.outer(backend),
        x,
        tx,
        new_contexts...,
    )
end
