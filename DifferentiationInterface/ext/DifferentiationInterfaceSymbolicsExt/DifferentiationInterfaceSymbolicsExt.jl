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

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

dense_ad(backend::AutoSymbolics) = backend
dense_ad(backend::AutoSparse{<:AutoSymbolics}) = ADTypes.dense_ad(backend)

variablize(::Number, name::Symbol) = variable(name)
variablize(x::AbstractArray, name::Symbol) = variables(name, axes(x)...)

function variablize(contexts::NTuple{C,DI.Context}) where {C}
    map(enumerate(contexts)) do (k, c)
        variablize(DI.unwrap(c), Symbol("context$k"))
    end
end

include("onearg.jl")
include("twoarg.jl")

end
