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

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

variablize(::Number, name::Symbol) = only(make_variables(name))
variablize(x::AbstractArray, name::Symbol) = make_variables(name, size(x)...)

function variablize(contexts::NTuple{C,DI.Context}) where {C}
    map(enumerate(contexts)) do (k, c)
        variablize(DI.unwrap(c), Symbol("context$k"))
    end
end

dense_ad(backend::AutoFastDifferentiation) = backend
dense_ad(backend::AutoSparse{<:AutoFastDifferentiation}) = ADTypes.dense_ad(backend)

myvec_unwrap(x) = myvec(DI.unwrap(x))

include("onearg.jl")
include("twoarg.jl")

end
