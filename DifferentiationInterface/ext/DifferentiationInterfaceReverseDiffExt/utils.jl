DI.overloaded_inputs(prep::ReverseDiffOneArgJacobianPrep) = eltype(prep.config.input)
DI.overloaded_inputs(prep::ReverseDiffTwoArgJacobianPrep) = eltype(prep.config.input)
