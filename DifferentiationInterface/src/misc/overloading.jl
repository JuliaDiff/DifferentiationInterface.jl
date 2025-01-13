"""
    overloaded_input_type(prep)

If it exists, return the overloaded input type which will be passed to the differentiated function when preparation result `prep` is reused.

!!! danger
    This function is experimental and not part of the public API.
"""
function overloaded_input_type end

## From prep alone

function overloaded_input_type(prep::PushforwardPullbackPrep)
    return overloaded_input_type(prep.pushforward_prep)
end

function overloaded_input_type(prep::PullbackPushforwardPrep)
    return overloaded_input_type(prep.pullback_prep)
end

function overloaded_input_type(prep::PushforwardDerivativePrep)
    return overloaded_input_type(prep.pushforward_prep)
end

function overloaded_input_type(prep::PullbackGradientPrep)
    return overloaded_input_type(prep.pullback_prep)
end

function overloaded_input_type(prep::PushforwardJacobianPrep)
    return overloaded_input_type(prep.pushforward_prep)
end

function overloaded_input_type(prep::PullbackJacobianPrep)
    return overloaded_input_type(prep.pullback_prep)
end
