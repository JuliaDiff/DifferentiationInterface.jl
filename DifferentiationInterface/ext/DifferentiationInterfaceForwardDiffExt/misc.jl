DI.overloaded_inputs(prep::ForwardDiffTwoArgPushforwardPrep) = eltype(prep.xdual_tmp)
DI.overloaded_inputs(prep::ForwardDiffOneArgJacobianPrep) = eltype(prep.config.duals[2])
DI.overloaded_inputs(prep::ForwardDiffTwoArgJacobianPrep) = eltype(prep.config.duals[2])