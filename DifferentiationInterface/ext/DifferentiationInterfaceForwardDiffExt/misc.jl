## Pushforward

function DI.overloaded_input(
    ::typeof(DI.pushforward),
    f::F,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, tx)
    return xdual
end

function DI.overloaded_input(
    ::typeof(DI.pushforward),
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    T = tag_type(f, backend, x)
    xdual = if x isa Number
        make_dual(T, x, tx)
    else
        make_dual_similar(T, x, tx)
    end
    return xdual
end

DI.overloaded_input_type(prep::ForwardDiffOneArgPushforwardPrep) = typeof(prep.xdual_tmp)
DI.overloaded_input_type(prep::ForwardDiffTwoArgPushforwardPrep) = typeof(prep.xdual_tmp)

## Derivative

function DI.overloaded_input_type(prep::ForwardDiffOneArgDerivativePrep)
    return DI.overloaded_input_type(prep.pushforward_prep)
end
function DI.overloaded_input_type(prep::ForwardDiffTwoArgDerivativePrep)
    return typeof(prep.config.duals)
end

## Gradient

DI.overloaded_input_type(prep::ForwardDiffGradientPrep) = typeof(prep.config.duals)

## Jacobian

function DI.overloaded_input_type(prep::ForwardDiffOneArgJacobianPrep)
    return typeof(prep.config.duals[2])
end
function DI.overloaded_input_type(prep::ForwardDiffTwoArgJacobianPrep)
    return typeof(prep.config.duals[2])
end
