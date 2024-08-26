"""
    basis(backend, a::AbstractArray, i::CartesianIndex)

Construct the `i`-th stardard basis array in the vector space of `a` with element type `eltype(a)`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
basis(::AbstractADType, a::AbstractArray, i) = basis(a, i)

function basis(a::AbstractArray{T,N}, i::CartesianIndex{N}) where {T,N}
    b = similar(a)
    copyto!(b, OneElement(one(T), Tuple(i), axes(a)))
    return b
end

function make_seed(x::AbstractArray, group::AbstractVector{<:Integer})
    seed = similar(x)
    seed .= zero(eltype(x))
    seed[group] .= one(eltype(x))
    return seed
end
