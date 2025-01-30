## Pushforward

function DI.overloaded_input_type(
    ::typeof(DI.pushforward),
    f::F,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    return DI.overloaded_input_type(DI.prepare_pushforward(f, backend, x, tx, contexts...))
end

function DI.overloaded_input_type(
    ::typeof(DI.pushforward),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    return DI.overloaded_input_type(
        DI.prepare_pushforward(f!, y, backend, x, tx, contexts...)
    )
end

DI.overloaded_input_type(prep::ForwardDiffOneArgPushforwardPrep) = typeof(prep.xdual_tmp)
DI.overloaded_input_type(prep::ForwardDiffTwoArgPushforwardPrep) = typeof(prep.xdual_tmp)

## Derivative

function DI.overloaded_input_type(
    ::typeof(DI.derivative),
    f::F,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.overloaded_input_type(DI.prepare_derivative(f, backend, x, contexts...))
end

function DI.overloaded_input_type(
    ::typeof(DI.derivative),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.overloaded_input_type(DI.prepare_derivative(f!, y, backend, x, contexts...))
end

function DI.overloaded_input_type(prep::ForwardDiffOneArgDerivativePrep)
    return DI.overloaded_input_type(prep.pushforward_prep)
end
DI.overloaded_input_type(prep::ForwardDiffTwoArgDerivativePrep) = typeof(prep.config.duals)

## Gradient

function DI.overloaded_input_type(
    ::typeof(DI.gradient), f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    return DI.overloaded_input_type(DI.prepare_gradient(f, backend, x, contexts...))
end

DI.overloaded_input_type(prep::ForwardDiffGradientPrep) = typeof(prep.config.duals)

## Jacobian

function DI.overloaded_input_type(
    ::typeof(DI.jacobian), f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    return DI.overloaded_input_type(DI.prepare_jacobian(f, backend, x, contexts...))
end

function DI.overloaded_input_type(
    ::typeof(DI.jacobian),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.overloaded_input_type(DI.prepare_jacobian(f!, y, backend, x, contexts...))
end

DI.overloaded_input_type(prep::ForwardDiffOneArgJacobianPrep) = typeof(prep.config.duals[2])
DI.overloaded_input_type(prep::ForwardDiffTwoArgJacobianPrep) = typeof(prep.config.duals[2])
