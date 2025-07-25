module DifferentiationInterfaceGPUArraysCoreExt

import DifferentiationInterface as DI
using GPUArraysCore: @allowscalar, AbstractGPUArray

function DI.basis(a::AbstractGPUArray{T}, i) where {T}
    b = similar(a)
    fill!(b, zero(T))
    @allowscalar b[i] = one(T)
    return b
end

function DI.multibasis(a::AbstractGPUArray{T}, inds) where {T}
    b = similar(a)
    fill!(b, zero(T))
    view(b, inds) .= one(T)
    return b
end

end
