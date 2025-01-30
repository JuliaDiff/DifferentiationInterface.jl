## Pushforward

function DI.overloaded_input_example(
    ::typeof(DI.pushforward),
    f::F,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    return DI.overloaded_input_example(
        DI.prepare_pushforward(f, backend, x, tx, contexts...)
    )
end

function DI.overloaded_input_example(
    ::typeof(DI.pushforward),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    return DI.overloaded_input_example(
        DI.prepare_pushforward(f!, y, backend, x, tx, contexts...)
    )
end

DI.overloaded_input_example(prep::ForwardDiffOneArgPushforwardPrep) = copy(prep.xdual_tmp)
DI.overloaded_input_example(prep::ForwardDiffTwoArgPushforwardPrep) = copy(prep.xdual_tmp)

## Derivative

function DI.overloaded_input_example(
    ::typeof(DI.derivative),
    f::F,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.overloaded_input_example(DI.prepare_derivative(f, backend, x, contexts...))
end

function DI.overloaded_input_example(
    ::typeof(DI.derivative),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.overloaded_input_example(
        DI.prepare_derivative(f!, y, backend, x, contexts...)
    )
end

function DI.overloaded_input_example(prep::ForwardDiffOneArgDerivativePrep)
    return DI.overloaded_input_example(prep.pushforward_prep)
end
DI.overloaded_input_example(prep::ForwardDiffTwoArgDerivativePrep) = copy(prep.config.duals)

## Gradient

function DI.overloaded_input_example(
    ::typeof(DI.gradient), f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    return DI.overloaded_input_example(DI.prepare_gradient(f, backend, x, contexts...))
end

DI.overloaded_input_example(prep::ForwardDiffGradientPrep) = copy(prep.config.duals)

## Jacobian

function DI.overloaded_input_example(
    ::typeof(DI.jacobian), f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    return DI.overloaded_input_example(DI.prepare_jacobian(f, backend, x, contexts...))
end

function DI.overloaded_input_example(
    ::typeof(DI.jacobian),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.overloaded_input_example(DI.prepare_jacobian(f!, y, backend, x, contexts...))
end

function DI.overloaded_input_example(prep::ForwardDiffOneArgJacobianPrep)
    return copy(prep.config.duals[2])
end
function DI.overloaded_input_example(prep::ForwardDiffTwoArgJacobianPrep)
    return copy(prep.config.duals[2])
end
