# Derivative
function DI.overloaded_inputs(prep::ForwardDiffOneArgDerivativePrep)
    return prep.pushforward_prep.xdual_tmp
end
DI.overloaded_inputs(prep::ForwardDiffTwoArgDerivativePrep) = prep.config.duals
# Gradient
DI.overloaded_inputs(prep::ForwardDiffGradientPrep) = prep.config.duals
# Jacobian
DI.overloaded_inputs(prep::ForwardDiffOneArgJacobianPrep) = prep.config.duals[2]
DI.overloaded_inputs(prep::ForwardDiffTwoArgJacobianPrep) = prep.config.duals[2]
