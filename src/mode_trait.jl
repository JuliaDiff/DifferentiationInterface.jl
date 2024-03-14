for operator in [
    :value_and_derivative,
    :value_and_multiderivative,
    :value_and_gradient,
    :value_and_jacobian,
    :derivative,
    :multiderivative,
    :gradient,
    :jacobian,
]
    @eval function $operator(
        backend::AbstractADType, f, x::Union{Number,AbstractArray}, extras=nothing
    )
        return $operator(backend, f, x, extras, autodiff_mode(backend))
    end
end

for operator in [
    :value_and_derivative!,
    :value_and_multiderivative!,
    :value_and_gradient!,
    :value_and_jacobian!,
    :derivative!,
    :multiderivative!,
    :gradient!,
    :jacobian!,
]
    @eval function $operator(
        storage::AbstractArray,
        backend::AbstractADType,
        f,
        x::Union{Number,AbstractArray},
        extras=nothing,
    )
        return $operator(storage, backend, f, x, extras, autodiff_mode(backend))
    end
end

for operator in [:value_and_multiderivative!, :value_and_jacobian!]
    @eval function $operator(
        y::AbstractArray,
        storage::AbstractArray,
        backend::AbstractADType,
        f!,
        x::Union{Number,AbstractArray},
        extras=nothing,
    )
        return $operator(y, storage, backend, f!, x, extras, autodiff_mode(backend))
    end
end
