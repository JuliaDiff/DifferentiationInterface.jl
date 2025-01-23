## Pushforward
DI.overloaded_input_type(prep::ForwardDiffOneArgPushforwardPrep) = typeof(prep.xdual_tmp)
DI.overloaded_input_type(prep::ForwardDiffTwoArgPushforwardPrep) = typeof(prep.xdual_tmp)

## Derivative
function DI.overloaded_input_type(prep::ForwardDiffOneArgDerivativePrep)
    return DI.overloaded_input_type(prep.pushforward_prep)
end
DI.overloaded_input_type(prep::ForwardDiffTwoArgDerivativePrep) = typeof(prep.config.duals)

## Gradient
DI.overloaded_input_type(prep::ForwardDiffGradientPrep) = typeof(prep.config.duals)

## Jacobian
DI.overloaded_input_type(prep::ForwardDiffOneArgJacobianPrep) = typeof(prep.config.duals[2])
DI.overloaded_input_type(prep::ForwardDiffTwoArgJacobianPrep) = typeof(prep.config.duals[2])
