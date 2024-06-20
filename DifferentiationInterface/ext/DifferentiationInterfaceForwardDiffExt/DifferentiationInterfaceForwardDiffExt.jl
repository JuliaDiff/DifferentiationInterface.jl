module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff
using Base: Fix1
using Compat
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    JacobianExtras,
    NoDerivativeExtras,
    NoSecondDerivativeExtras,
    PushforwardExtras
using ForwardDiff.DiffResults:
    DiffResults, DiffResult, GradientResult, HessianResult, MutableDiffResult
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    ForwardDiff,
    GradientConfig,
    HessianConfig,
    JacobianConfig,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: dot, mul!

DI.check_available(::AutoForwardDiff) = true

function DI.pick_batchsize(::AutoForwardDiff{C}, dimension::Integer) where {C}
    if isnothing(C)
        return ForwardDiff.pickchunksize(dimension)
    else
        return min(dimension, C)
    end
end

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")

end # module
