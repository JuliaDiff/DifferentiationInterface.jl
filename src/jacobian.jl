const JAC_NOTES = """
## Notes

Regardless of the shape of `x` and `y`, if `x` has length `n` and `y` has length `m`, then `jac` is expected to be a `m × n` matrix.
This function acts as if the input and output had been flattened with `vec`.
"""

function check_jac(jac::AbstractMatrix, x::AbstractArray, y::AbstractArray)
    nx, ny = length(x), length(y)
    size(jac) != (ny, nx) && throw(
        DimensionMismatch("Size of Jacobian buffer doesn't match expected size ($ny, $nx)"),
    )
    return nothing
end

"""
    value_and_jacobian!(jac, backend, f, x, [extras]) -> (y, jac)
    value_and_jacobian!(y, jac, backend, f!, x, [extras]) -> (y, jac)

Compute the primal value `y = f(x)` and the Jacobian matrix `jac = ∂f(x)` of an array-to-array function, overwriting `jac`.

$JAC_NOTES
"""
function value_and_jacobian!(
    jac::AbstractMatrix,
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_jacobian(backend, f, x),
) where {F}
    return value_and_jacobian_aux!(
        jac, backend, f, x, extras, supports_pushforward(backend)
    )
end

function value_and_jacobian!(
    y::AbstractArray,
    jac::AbstractMatrix,
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_jacobian(backend, f, x, y),
) where {F}
    return value_and_jacobian_aux!(
        y, jac, backend, f, x, extras, supports_pushforward(backend)
    )
end

function value_and_jacobian_aux!(
    jac, backend, f::F, x, extras, ::PushforwardSupported
) where {F}
    y = f(x)
    check_jac(jac, x, y)
    for (k, j) in enumerate(eachindex(IndexCartesian(), x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        pushforward!(jac_col_j, backend, f, x, dx_j, extras)
    end
    return y, jac
end

function value_and_jacobian_aux!(
    y, jac, backend, f!::F, x, extras, ::PushforwardSupported
) where {F}
    check_jac(jac, x, y)
    for (k, j) in enumerate(eachindex(IndexCartesian(), x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        value_and_pushforward!(y, jac_col_j, backend, f!, x, dx_j, extras)
    end
    return y, jac
end

function value_and_jacobian_aux!(
    jac, backend, f::F, x, extras, ::PushforwardNotSupported
) where {F}
    y = f(x)
    check_jac(jac, x, y)
    for (k, i) in enumerate(eachindex(IndexCartesian(), y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        pullback!(jac_row_i, backend, f, x, dy_i, extras)
    end
    return y, jac
end

function value_and_jacobian_aux!(
    y, jac, backend, f!::F, x, extras, ::PushforwardNotSupported
) where {F}
    check_jac(jac, x, y)
    for (k, i) in enumerate(eachindex(IndexCartesian(), y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        value_and_pullback!(y, jac_row_i, backend, f!, x, dy_i, extras)
    end
    return y, jac
end

"""
    value_and_jacobian(backend, f, x, [extras]) -> (y, jac)

Compute the primal value `y = f(x)` and the Jacobian matrix `jac = ∂f(x)` of an array-to-array function.

$JAC_NOTES
"""
function value_and_jacobian(
    backend::AbstractADType, f::F, x::AbstractArray, extras=prepare_jacobian(backend, f, x)
) where {F}
    y = f(x)
    T = promote_type(eltype(x), eltype(y))
    jac = similar(y, T, length(y), length(x))
    return value_and_jacobian!(jac, backend, f, x, extras)
end

"""
    jacobian!(jac, backend, f, x, [extras]) -> jac

Compute the Jacobian matrix `jac = ∂f(x)` of an array-to-array function, overwriting `jac`.

$JAC_NOTES
"""
function jacobian!(
    jac::AbstractMatrix,
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_jacobian(backend, f, x),
) where {F}
    return last(value_and_jacobian!(jac, backend, f, x, extras))
end

"""
    jacobian(backend, f, x, [extras]) -> jac

Compute the Jacobian matrix `jac = ∂f(x)` of an array-to-array function.

$JAC_NOTES
"""
function jacobian(
    backend::AbstractADType, f::F, x::AbstractArray, extras=prepare_jacobian(backend, f, x)
) where {F}
    return last(value_and_jacobian(backend, f, x, extras))
end
