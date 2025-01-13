"""
    overloaded_input_type(prep)
"""
function overloaded_input_type end

"""
    overloaded_input_type_pushforward(f, backend, x, tx, contexts...)
    overloaded_input_type_pushforward(f!, y, backend, x, tx, contexts...)
"""
function overloaded_input_type_pushforward end

"""
    overloaded_input_type_pushforward(f, backend, x, ty, contexts...)
    overloaded_input_type_pushforward(f!, y, backend, x, ty, contexts...)
"""
function overloaded_input_type_pullback end

"""
    overloaded_input_type_derivative(f, backend, x, contexts...)
    overloaded_input_type_derivative(f!, y, backend, x, contexts...)
"""
function overloaded_input_type_derivative end

"""
    overloaded_input_type_gradient(f, backend, x, contexts...)
"""
function overloaded_input_type_gradient end

"""
    overloaded_input_type_jacobian(f, backend, x, contexts...)
    overloaded_input_type_jacobian(f!, y, backend, x, contexts...)
"""
function overloaded_input_type_jacobian end

## Fallback on prep

function overloaded_input_type_pushforward(
    f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_pushforward(f, backend, x, tx, contexts...))
end
function overloaded_input_type_pushforward(
    f!::F, y, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_pushforward(f!, y, backend, x, tx, contexts...))
end

function overloaded_input_type_pullback(
    f::F, backend::AbstractADType, x, ty::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_pullback(f, backend, x, ty, contexts...))
end
function overloaded_input_type_pullback(
    f!::F, y, backend::AbstractADType, x, ty::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_pullback(f!, y, backend, x, ty, contexts...))
end

function overloaded_input_type_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_derivative(f, backend, x, contexts...))
end
function overloaded_input_type_derivative(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_derivative(f!, y, backend, x, contexts...))
end

function overloaded_input_type_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_derivative(f, backend, x, contexts...))
end
function overloaded_input_type_derivative(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_derivative(f!, y, backend, x, contexts...))
end

function overloaded_input_type_gradient(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_gradient(f, backend, x, contexts...))
end

function overloaded_input_type_jacobian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_jacobian(f, backend, x, contexts...))
end
function overloaded_input_type_jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return overloaded_input_type(prepare_jacobian(f!, y, backend, x, contexts...))
end

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
