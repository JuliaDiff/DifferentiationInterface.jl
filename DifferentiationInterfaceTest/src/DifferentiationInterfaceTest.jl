"""
    DifferentiationInterfaceTest

Testing and benchmarking utilities for automatic differentiation in Julia.
"""
module DifferentiationInterfaceTest

using ADTypes:
    ADTypes,
    AbstractADType,
    AbstractMode,
    AutoSparse,
    ForwardMode,
    ForwardOrReverseMode,
    ReverseMode,
    SymbolicMode,
    mode
using AllocCheck: check_allocs
using DataAPI: DataAPI
import DifferentiationInterface as DI
using DifferentiationInterface:
    prepare_pushforward,
    prepare_pushforward_same_point,
    prepare!_pushforward,
    pushforward,
    pushforward!,
    value_and_pushforward,
    value_and_pushforward!,
    prepare_pullback,
    prepare_pullback_same_point,
    prepare!_pullback,
    pullback,
    pullback!,
    value_and_pullback,
    value_and_pullback!,
    prepare_derivative,
    prepare!_derivative,
    derivative,
    derivative!,
    value_and_derivative,
    value_and_derivative!,
    prepare_gradient,
    prepare!_gradient,
    gradient,
    gradient!,
    value_and_gradient,
    value_and_gradient!,
    prepare_jacobian,
    prepare!_jacobian,
    jacobian,
    jacobian!,
    value_and_jacobian,
    value_and_jacobian!,
    prepare_second_derivative,
    prepare!_second_derivative,
    second_derivative,
    second_derivative!,
    value_derivative_and_second_derivative,
    value_derivative_and_second_derivative!,
    prepare_hvp,
    prepare_hvp_same_point,
    prepare!_hvp,
    hvp,
    hvp!,
    gradient_and_hvp,
    gradient_and_hvp!,
    prepare_hessian,
    prepare!_hessian,
    hessian,
    hessian!,
    value_gradient_and_hessian,
    value_gradient_and_hessian!
using DifferentiationInterface:
    DerivativePrep,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    PullbackPrep,
    PushforwardPrep,
    SecondDerivativePrep
using DifferentiationInterface:
    SecondOrder,
    MixedMode,
    inner,
    mode,
    outer,
    inplace_support,
    pushforward_performance,
    pullback_performance,
    check_available,
    forward_counterpart,
    reverse_counterpart
using DifferentiationInterface: Rewrap, Context, Constant, Cache, ConstantOrCache, unwrap
using DifferentiationInterface: PreparationMismatchError
using DocStringExtensions: TYPEDFIELDS, TYPEDSIGNATURES
using Functors: fmap
using LinearAlgebra: Adjoint, Diagonal, Transpose, I, dot, parent
using PrecompileTools: @compile_workload
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays:
    SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, nnz, sparse, spdiagm
using Tables: Tables, AbstractRow, AbstractColumns
using Test: @testset, @test, @test_throws

include("basics/utils.jl")
include("basics/operators.jl")
include("basics/result.jl")
include("basics/scenario.jl")

include("tests/loop.jl")
include("tests/correctness.jl")
include("tests/benchmark.jl")

export Scenario, Result, TripleScenario
export test_differentiation

end
