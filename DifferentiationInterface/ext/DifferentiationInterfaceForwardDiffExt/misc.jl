DI.overloaded_inputs(prep::ForwardDiffTwoArgPushforwardPrep) = prep.xdual_tmp
DI.overloaded_inputs(prep::ForwardDiffOneArgJacobianPrep) = prep.config.duals[2]
DI.overloaded_inputs(prep::ForwardDiffTwoArgJacobianPrep) = prep.config.duals[2]
