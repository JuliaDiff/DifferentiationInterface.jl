## Pushforward

function DI.value_and_pushforward!(
    _dy::Real, ::AutoForwardDiff, f, x::Real, dx, extras::Nothing
)
    T = tag_type(f, x)
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::AbstractArray, ::AutoForwardDiff, f, x::Real, dx, extras::Nothing
)
    T = tag_type(f, x)
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

function DI.value_and_pushforward!(
    _dy::Real, ::AutoForwardDiff, f, x::AbstractArray, dx, extras::Nothing
)
    T = tag_type(f, x)
    xdual = Dual{T}.(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::AbstractArray, ::AutoForwardDiff, f, x::AbstractArray, dx, extras::Nothing
)
    T = tag_type(f, x)
    xdual = Dual{T}.(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

## Derivative

function DI.derivative(::AutoForwardDiff, f, x::Number, extras::Nothing)
    return derivative(f, x)
end

## Multiderivative

function DI.multiderivative!(
    multider::AbstractArray, ::AutoForwardDiff, f, x::Number, extras::Nothing
)
    derivative!(multider, f, x)
    return multider
end

function DI.multiderivative(::AutoForwardDiff, f, x::Number, extras::Nothing)
    return derivative(f, x)
end

## Gradient

function DI.prepare_gradient(backend::AutoForwardDiff, f, x::AbstractArray)
    return GradientConfig(f, x, choose_chunk(backend, x))
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoForwardDiff, f, x::AbstractArray, config::GradientConfig
)
    result = DiffResults.DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, f, x, config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    ::AutoForwardDiff, f, x::AbstractArray, config::GradientConfig
)
    result = DiffResults.GradientResult(x)
    result = gradient!(result, f, x, config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.gradient!(
    grad::AbstractArray, ::AutoForwardDiff, f, x::AbstractArray, config::GradientConfig
)
    gradient!(grad, f, x, config)
    return grad
end

function DI.gradient(::AutoForwardDiff, f, x::AbstractArray, config::GradientConfig)
    return gradient(f, x, config)
end

## Jacobian

function DI.prepare_jacobian(backend::AutoForwardDiff, f, x::AbstractArray)
    return JacobianConfig(f, x, choose_chunk(backend, x))
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, ::AutoForwardDiff, f, x::AbstractArray, config::JacobianConfig
)
    y = f(x)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, f, x, config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    ::AutoForwardDiff, f, x::AbstractArray, config::JacobianConfig
)
    return f(x), jacobian(f, x, config)
end

function DI.jacobian!(
    jac::AbstractMatrix, ::AutoForwardDiff, f, x::AbstractArray, config::JacobianConfig
)
    jacobian!(jac, f, x, config)
    return jac
end

function DI.jacobian(::AutoForwardDiff, f, x::AbstractArray, config::JacobianConfig)
    return jacobian(f, x, config)
end
