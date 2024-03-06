"""
    basisarray(a::AbstractArray, i)
    basisarray(backend, a::AbstractArray, i)

Construct the `i`-th stardard basis array in the vector space of `a` with element type `eltype(v)`.

## Note

If an AD backend benefits from a more specialized unit vector implementation,
this function can be extended on the backend type.

function basisarray(::AbstractBackend, a::AbstractVector{T}, i::Integer) where {T}
    return OneElement(one(T), i, length(v))
end
"""
basisarray(::AbstractBackend, a::AbstractArray, i::Integer) = basisarray(a, i)

function basisarray(a::AbstractArray{T}, i::Vararg{<:Integer,N}) where {T,N}
    return OneElement(one(T), i, axes(a))
end
