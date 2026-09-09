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

dense_ad(backend::AutoSymbolics) = backend
dense_ad(backend::AutoSparse{<:AutoSymbolics}) = ADTypes.dense_ad(backend)

variablize(::Number, name::Symbol) = variable(name)
variablize(x::AbstractArray, name::Symbol) = variables(name, axes(x)...)
variablize(::T, name::Symbol) where {T} = variable(name; T)

#=
The value of a `PrepTimeConstant` is fixed at preparation, so it is substituted directly into the
symbolic expression instead of becoming a variable. This lets the constant participate in symbolic
simplification of the derivative, which is the whole point of the context type.
=#
function variablize(contexts::NTuple{C, DI.Context}) where {C}
    return ntuple(Val(C)) do k
        c = contexts[k]
        if c isa DI.PrepTimeConstant
            DI.unwrap(c)
        else
            variablize(DI.unwrap(c), Symbol("context$k"))
        end
    end
end

#=
The generated function still accepts one argument per context, so that execution does not need to
know which contexts were folded in. The arguments standing for a `PrepTimeConstant` are fresh
variables which do not occur in the expression, and are therefore ignored at run time.
=#
function argumentize(
        context_vars::NTuple{C}, contexts::NTuple{C, DI.Context}
    ) where {C}
    return ntuple(Val(C)) do k
        c = contexts[k]
        if c isa DI.PrepTimeConstant
            variablize(DI.unwrap(c), Symbol("unused_context$k"))
        else
            context_vars[k]
        end
    end
end

function erase_cache_vars!(
        context_vars::NTuple{C}, contexts::NTuple{C, DI.Context}
    ) where {C}
    # erase the active data from caches before building function
    for (v, c) in zip(context_vars, contexts)
        if c isa DI.Cache
            fill!(v, zero(eltype(v)))
        end
    end
    return
end

include("onearg.jl")
include("twoarg.jl")

end
