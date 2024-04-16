module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface: AutoFastDifferentiation, AutoSparseFastDifferentiation
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    SecondDerivativeExtras
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
using FillArrays: Fill
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

const AnyAutoFastDifferentiation = Union{
    AutoFastDifferentiation,AutoSparseFastDifferentiation
}

DI.check_available(::AutoFastDifferentiation) = true
DI.mode(::AutoFastDifferentiation) = ADTypes.AbstractSymbolicDifferentiationMode
DI.pushforward_performance(::AutoFastDifferentiation) = DI.PushforwardFast()
DI.pullback_performance(::AutoFastDifferentiation) = DI.PullbackSlow()

monovec(x::Number) = Fill(x, 1)

myvec(x::Number) = monovec(x)
myvec(x::AbstractArray) = vec(x)

issparse(::AutoFastDifferentiation) = false
issparse(::AutoSparseFastDifferentiation) = true

include("onearg.jl")
include("twoarg.jl")

end
