## Pullback

function DI.overloaded_input(
    ::typeof(DI.pullback),
    f,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return copy(ReverseDiff.track(x))
end

function DI.overloaded_input(
    ::typeof(DI.pullback),
    f!,
    y,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    return copy(ReverseDiff.track(x))
end

## Gradient
DI.overloaded_input_type(prep::ReverseDiffGradientPrep) = typeof(prep.config.input)

## Jacobian
DI.overloaded_input_type(prep::ReverseDiffOneArgJacobianPrep) = typeof(prep.config.input)
DI.overloaded_input_type(prep::ReverseDiffTwoArgJacobianPrep) = typeof(prep.config.input)
