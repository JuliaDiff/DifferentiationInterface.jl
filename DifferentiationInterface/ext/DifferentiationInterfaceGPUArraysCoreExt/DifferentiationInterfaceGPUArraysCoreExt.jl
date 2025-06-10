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
    for i in inds
        @allowscalar b[i] = one(T)
    end
    return b
end

end
