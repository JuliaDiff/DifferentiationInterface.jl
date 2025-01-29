struct ForwardDiffOverSomethingHVPPrep{E1<:DI.GradientPrep,E2<:DI.PushforwardPrep} <:
       DI.HVPPrep
    inner_gradient_prep::E1
    outer_pushforward_prep::E2
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
    outer_pushforward_prep = DI.prepare_pushforward(
        DI.shuffled_gradient, DI.outer(backend), x, tx, new_contexts...
    )
    return ForwardDiffOverSomethingHVPPrep(inner_gradient_prep, outer_pushforward_prep)
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
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    return DI.pushforward!(
        DI.shuffled_gradient,
        tg,
        outer_pushforward_prep,
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
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = DI.Rewrap(contexts...)
    new_contexts = (
        DI.FunctionContext(f),
        PrepContext(inner_gradient_prep),
        DI.BackendContext(DI.inner(backend)),
        DI.Constant(rewrap),
        contexts...,
    )
    new_grad, _ = DI.value_and_pushforward!(
        DI.shuffled_gradient,
        tg,
        outer_pushforward_prep,
        DI.outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return copyto!(grad, new_grad), tg
end
