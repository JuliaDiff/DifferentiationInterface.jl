include("../../testutils.jl")

using DifferentiationInterface
import DifferentiationInterface as DI
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
import Enzyme, Reactant
using Test

backend = AutoReactant()

@test check_available(backend)
@test check_inplace(backend)

scen1 = DIT.Scenario(

)

test_differentiation(
    backend, DifferentiationInterfaceTest.default_scenarios(;
        include_constantified = false, include_cachified = false
    );
    excluded = vcat(SECOND_ORDER, :jacobian, :derivative, :pushforward, :pullback),
    logging = false
)
