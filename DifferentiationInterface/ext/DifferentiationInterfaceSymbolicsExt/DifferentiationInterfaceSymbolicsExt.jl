module DifferentiationInterfaceSymbolicsExt

using ADTypes: ADTypes, AutoSymbolics, AutoSparse
import DifferentiationInterface as DI
using LinearAlgebra: dot
using Symbolics:
    build_function,
    derivative,
    gradient,
    hessian,
    hessian_sparsity,
    jacobian,
    jacobian_sparsity,
    sparsehessian,
    sparsejacobian,
    substitute,
    variable,
    variables
using Symbolics.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.check_available(::AutoSymbolics) = true
DI.pullback_performance(::AutoSymbolics) = DI.PullbackSlow()

monovec(x::Number) = [x]

myvec(x::Number) = monovec(x)
myvec(x::AbstractArray) = vec(x)

dense_ad(backend::AutoSymbolics) = backend
dense_ad(backend::AutoSparse{<:AutoSymbolics}) = ADTypes.dense_ad(backend)

include("onearg.jl")
include("twoarg.jl")

end
