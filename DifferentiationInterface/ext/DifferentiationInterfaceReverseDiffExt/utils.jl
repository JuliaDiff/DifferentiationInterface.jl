## Gradient
DI.overloaded_input_type(prep::ReverseDiffGradientPrep) = typeof(prep.config.input)

## Jacobian
DI.overloaded_input_type(prep::ReverseDiffOneArgJacobianPrep) = typeof(prep.config.input)
DI.overloaded_input_type(prep::ReverseDiffTwoArgJacobianPrep) = typeof(prep.config.input)
