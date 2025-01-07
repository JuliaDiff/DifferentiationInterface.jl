# Gradient
DI.overloaded_inputs(prep::ReverseDiffGradientPrep) = prep.config.input
# Jacobian
DI.overloaded_inputs(prep::ReverseDiffOneArgJacobianPrep) = prep.config.input
DI.overloaded_inputs(prep::ReverseDiffTwoArgJacobianPrep) = prep.config.input
