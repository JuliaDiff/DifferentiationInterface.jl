module DifferentiationInterfaceGPUArraysCoreExt

using Adapt: adapt
import DifferentiationInterface as DI
using GPUArraysCore: @allowscalar, AbstractGPUArray

function DI.basis(a::AbstractGPUArray{T}, i) where {T}
    b = similar(a)
    fill!(b, zero(T))
    @allowscalar b[i] = oneunit(T)
    return b
end

function DI.multibasis(a::AbstractGPUArray{T}, inds) where {T}
    b = similar(a)
    fill!(b, zero(T))
    view(b, inds) .= oneunit(T)
    return b
end

function DI.arroftup_to_tupofarr(
        tx::AbstractArray{<:NTuple{B, <:Number}}, x::AbstractGPUArray{<:Number}
    ) where {B}
    return ntuple(b -> adapt(typeof(x), getindex.(tx, b)), Val(B))
end

end
