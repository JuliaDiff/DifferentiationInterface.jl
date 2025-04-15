module DifferentiationInterfaceGPUArraysCoreExt

import DifferentiationInterface as DI
using GPUArraysCore: AbstractGPUArray

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
    return ifelse(ind == oe.ind, oe.val, zero(eltype(oe.a)))
end

function DI.basis(a::AbstractGPUArray{T}, i) where {T}
    b = zero(a)
    b .+= OneElement(i, one(T), a)
    return b
end

end
