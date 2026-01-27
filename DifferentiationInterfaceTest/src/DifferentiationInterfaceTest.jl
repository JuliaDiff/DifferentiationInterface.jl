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
    pullback_performance
using DifferentiationInterface: Rewrap, Context, Constant, Cache, ConstantOrCache, unwrap
using DifferentiationInterface: PreparationMismatchError
using DocStringExtensions: TYPEDFIELDS, TYPEDSIGNATURES
using LinearAlgebra: Adjoint, Diagonal, Transpose, I, dot, parent
using PrecompileTools: @compile_workload
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays:
    SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, nnz, sparse, spdiagm
using Tables: Tables, AbstractRow, AbstractColumns
using Test: @testset, @test, @test_throws

"""
    FIRST_ORDER = [:pushforward, :pullback, :derivative, :gradient, :jacobian]

List of all first-order operators, to facilitate exclusion during tests.
"""
const FIRST_ORDER = [:pushforward, :pullback, :derivative, :gradient, :jacobian]

"""
    SECOND_ORDER = [:hvp, :second_derivative, :hessian]

List of all second-order operators, to facilitate exclusion during tests.
"""
const SECOND_ORDER = [:hvp, :second_derivative, :hessian]

const ALL_OPS = (
    :pushforward,
    :pullback,
    :derivative,
    :gradient,
    :jacobian,
    :hvp,
    :second_derivative,
    :hessian,
)

include("utils.jl")

include("scenarios/scenario.jl")
include("scenarios/modify.jl")
include("scenarios/default.jl")
include("scenarios/sparse.jl")
include("scenarios/complex.jl")
include("scenarios/allocfree.jl")
include("scenarios/empty.jl")
include("scenarios/extensions.jl")

include("tests/correctness_eval.jl")
include("tests/prep_eval.jl")
include("tests/type_stability.jl")
include("tests/benchmark.jl")
include("tests/allocs_eval.jl")

include("test_differentiation.jl")

export FIRST_ORDER, SECOND_ORDER
export Scenario, compute_results
export test_differentiation, benchmark_differentiation
export DifferentiationBenchmarkDataRow

# @compile_workload begin
#     default_scenarios(; include_constantified = true, include_cachified = true)
# end

end
