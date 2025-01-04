struct OneElement{I,N,T,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    ind::I
    val::T
    a::A

    function OneElement(ind::Integer, val::T, a::A) where {N,T,A<:AbstractArray{T,N}}
        right_ind = eachindex(a)[ind]
        return new{typeof(right_ind),N,T,A}(right_ind, val, a)
    end

    function OneElement(
        ind::CartesianIndex{N}, val::T, a::A
    ) where {N,T,A<:AbstractArray{T,N}}
        linear_ind = LinearIndices(a)[ind]
        right_ind = eachindex(a)[linear_ind]
        return new{typeof(right_ind),N,T,A}(right_ind, val, a)
    end
end

Base.size(oe::OneElement) = size(oe.a)
Base.IndexStyle(oe::OneElement) = Base.IndexStyle(oe.a)

function Base.getindex(oe::OneElement{<:Integer}, ind::Integer)
    if ind == oe.ind
        return oe.val
    else
        return zero(eltype(oe.a))
    end
end

function Base.getindex(oe::OneElement{<:CartesianIndex{N}}, ind::Vararg{Int,N}) where {N}
    if ind == Tuple(oe.ind)
        return oe.val
    else
        return zero(eltype(oe.a))
    end
end

"""
    basis(backend, a::AbstractArray, i)

Construct the `i`-th standard basis array in the vector space of `a` with element type `eltype(a)`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
basis(::AbstractADType, a::AbstractArray{<:Real}, i) = basis(a, i)

"""
    multibasis(backend, a::AbstractArray, inds::AbstractVector)

Construct the sum of the `i`-th standard basis arrays in the vector space of `a` with element type `eltype(a)`, for all `i ∈ inds`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
multibasis(::AbstractADType, a::AbstractArray{<:Real}, inds) = multibasis(a, inds)

function basis(a::AbstractArray{T,N}, i) where {T<:Real,N}
    return zero(a) + OneElement(i, one(T), a)
end

function multibasis(a::AbstractArray{T,N}, inds::AbstractVector) where {T<:Real,N}
    seed = zero(a)
    for i in inds
        seed += OneElement(i, one(T), a)
    end
    return seed
end

realone(x::Real) = one(x)
