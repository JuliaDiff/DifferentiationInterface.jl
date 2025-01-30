## Pullback

function DI.overloaded_input_example(
    ::typeof(DI.pullback),
    f,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return copy(ReverseDiff.track(x))
end

## Gradient
DI.overloaded_input_example(prep::ReverseDiffGradientPrep) = copy(prep.config.input)

## Jacobian
DI.overloaded_input_example(prep::ReverseDiffOneArgJacobianPrep) = copy(prep.config.input)
DI.overloaded_input_example(prep::ReverseDiffTwoArgJacobianPrep) = copy(prep.config.input)
