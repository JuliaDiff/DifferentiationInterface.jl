module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes, AutoFastDifferentiation, AutoSparse
import DifferentiationInterface as DI
using FastDifferentiation:
    derivative,
    hessian,
    hessian_times_v,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables,
    sparse_hessian,
    sparse_jacobian
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.check_available(::AutoFastDifferentiation) = true

monovec(x::Number) = [x]

myvec(x::Number) = monovec(x)
myvec(x::AbstractArray) = vec(x)

dense_ad(backend::AutoFastDifferentiation) = backend
dense_ad(backend::AutoSparse{<:AutoFastDifferentiation}) = ADTypes.dense_ad(backend)

include("onearg.jl")
include("twoarg.jl")

end
