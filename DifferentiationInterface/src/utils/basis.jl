"""
    OneElement

Efficient storage for a one-hot array, aka an array in the standard Euclidean basis.
"""
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
    basis(a::AbstractArray, i)

Construct the `i`-th standard basis array in the vector space of `a`.
"""
function basis(a::AbstractArray{T}, i) where {T}
    b = similar(a)
    fill!(b, zero(T))
    b .+= OneElement(i, one(T), a)
    if ismutable_array(a)
        return b
    else
        return map(+, zero(a), b)
    end
end

"""
    multibasis(a::AbstractArray, inds)

Construct the sum of the `i`-th standard basis arrays in the vector space of `a` for all `i ∈ inds`.

!!! warning
    Does not work on GPU, since this is intended for sparse autodiff and SparseMatrixColorings.jl doesn't work on GPUs either.
"""
function multibasis(a::AbstractArray{T}, inds) where {T}
    b = similar(a)
    fill!(b, zero(T))
    for i in inds
        b[i] = one(T)
    end
    return ismutable_array(a) ? b : map(+, zero(a), b)
end
