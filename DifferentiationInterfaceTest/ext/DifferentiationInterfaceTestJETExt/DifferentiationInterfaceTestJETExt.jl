module DifferentiationInterfaceTestJETExt

using ADTypes: AbstractADType
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
using DifferentiationInterfaceTest: ALL_OPS, Scenario, mysimilar
import DifferentiationInterfaceTest as DIT
using JET: @test_opt

include("type_stability_eval.jl")

end
