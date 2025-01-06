module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
using Base: Fix2
import DifferentiationInterface as DI
import DiffResults as DR
using DiffResults: DiffResults, DiffResult, GradientResult, HessianResult, MutableDiffResult
using LinearAlgebra: dot, mul!
using ReverseDiff:
    ReverseDiff,
    CompiledGradient,
    CompiledHessian,
    CompiledJacobian,
    GradientConfig,
    GradientTape,
    HessianConfig,
    HessianTape,
    JacobianConfig,
    JacobianTape,
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!

DI.check_available(::AutoReverseDiff) = true

function DI.basis(::AutoReverseDiff, a::AbstractArray{T}, i) where {T}
    return DI.OneElement(i, one(T), a)
end

include("onearg.jl")
include("twoarg.jl")

end # module
