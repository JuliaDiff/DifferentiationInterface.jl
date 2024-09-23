"""
    SecondOrder

Combination of two backends for second-order differentiation.

!!! danger
    `SecondOrder` backends do not support first-order operators.

# Constructor

    SecondOrder(outer_backend, inner_backend)

# Fields

- `outer::AbstractADType`: backend for the outer differentiation
- `inner::AbstractADType`: backend for the inner differentiation
"""
struct SecondOrder{ADO<:AbstractADType,ADI<:AbstractADType} <: AbstractADType
    outer::ADO
    inner::ADI
end

function Base.show(io::IO, backend::SecondOrder)
    return print(
        io,
        SecondOrder,
        "(",
        repr(outer(backend); context=io),
        ", ",
        repr(inner(backend); context=io),
        ")",
    )
end

"""
    inner(backend::SecondOrder)

Return the inner backend of a [`SecondOrder`](@ref) object, tasked with differentiation at the first order.
"""
inner(backend::SecondOrder) = backend.inner

"""
    outer(backend::SecondOrder)

Return the outer backend of a [`SecondOrder`](@ref) object, tasked with differentiation at the second order.
"""
outer(backend::SecondOrder) = backend.outer

"""
    mode(backend::SecondOrder)

Return the _outer_ mode of the second-order backend.
"""
ADTypes.mode(backend::SecondOrder) = mode(outer(backend))

"""
    nested(backend)

Return a possibly modified `backend` that can work while nested inside another differentiation procedure.
"""
nested(backend::AbstractADType) = backend
