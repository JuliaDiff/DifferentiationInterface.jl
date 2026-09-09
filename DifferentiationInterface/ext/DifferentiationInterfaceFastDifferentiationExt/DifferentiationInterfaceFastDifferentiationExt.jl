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

#=
The value of a `PrepTimeConstant` is fixed at preparation, so it is substituted directly into the
symbolic expression instead of becoming a variable. This lets the constant participate in symbolic
simplification of the derivative, which is the whole point of the context type.
=#
function variablize(contexts::NTuple{C, DI.Context}) where {C}
    return map(enumerate(contexts)) do (k, c)
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
function argumentize(context_vars, contexts::NTuple{C, DI.Context}) where {C}
    return map(enumerate(contexts)) do (k, c)
        if c isa DI.PrepTimeConstant
            variablize(DI.unwrap(c), Symbol("unused_context$k"))
        else
            context_vars[k]
        end
    end
end

dense_ad(backend::AutoFastDifferentiation) = backend
dense_ad(backend::AutoSparse{<:AutoFastDifferentiation}) = ADTypes.dense_ad(backend)

myvec_unwrap(x) = myvec(DI.unwrap(x))

include("onearg.jl")
include("twoarg.jl")

end
