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
    inner(backend::AbstractADType)

Return the inner backend of a [`SecondOrder`](@ref) object, tasked with differentiation at the first order.

For any other backend type, this function acts like the identity.
"""
inner(backend::SecondOrder) = backend.inner
inner(backend::AbstractADType) = backend

"""
    outer(backend::SecondOrder)
    outer(backend::AbstractADType)

Return the outer backend of a [`SecondOrder`](@ref) object, tasked with differentiation at the second order.

For any other backend type, this function acts like the identity.
"""
outer(backend::SecondOrder) = backend.outer
outer(backend::AbstractADType) = backend

"""
    mode(backend::SecondOrder)

Return the _outer_ mode of the second-order backend.
"""
ADTypes.mode(backend::SecondOrder) = mode(outer(backend))
