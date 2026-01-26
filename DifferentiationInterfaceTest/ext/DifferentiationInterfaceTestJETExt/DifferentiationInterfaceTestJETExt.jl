module DifferentiationInterfaceTestJETExt

using ADTypes: AbstractADType
using DifferentiationInterfaceTest: ALL_OPS, Scenario
import DifferentiationInterfaceTest as DIT
using JET: @test_opt

include("type_stability_eval.jl")

end
