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
    jacobian!,
    value

DI.check_available(::AutoReverseDiff) = true
DI.check_operator_overloading(::AutoReverseDiff) = true

include("onearg.jl")
include("twoarg.jl")
include("utils.jl")

end # module
