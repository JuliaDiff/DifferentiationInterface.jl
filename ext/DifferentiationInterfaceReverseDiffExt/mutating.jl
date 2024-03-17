## Jacobian

function DI.prepare_jacobian(
    backend::AutoReverseDiff, f!, x::AbstractArray, y::AbstractArray
)
    tape = JacobianTape(f!, y, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.value_and_jacobian!(
    y::AbstractArray,
    jac::AbstractMatrix,
    backend::AutoReverseDiff,
    f!,
    x::AbstractArray,
    extras::Nothing,
)
    return DI.value_and_jacobian!(
        y, jac, backend, f!, x, DI.prepare_jacobian(backend, f!, x, y)
    )
end

function DI.value_and_jacobian!(
    y::AbstractArray,
    jac::AbstractMatrix,
    backend::AutoReverseDiff,
    f!,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

## Pullback

function DI.value_and_pullback!(
    y::AbstractArray,
    dx::AbstractArray,
    backend::AutoReverseDiff,
    f!,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
)
    jac = similar(dy, length(y), length(x))  # allocates
    y, jac = DI.value_and_jacobian!(y, jac, backend, f!, x, extras)
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback!(
    y::AbstractArray,
    _dx::Number,
    backend::AutoReverseDiff,
    f!,
    x::Number,
    dy,
    extras::Nothing,
)
    x_array = [x]
    dx_array = similar(x_array)
    f!_only(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    y, dx_array = DI.value_and_pullback!(y, dx_array, backend, f!_only, x_array, dy, extras)
    return y, only(dx_array)
end
