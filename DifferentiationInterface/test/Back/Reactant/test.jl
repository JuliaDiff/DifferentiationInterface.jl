using Pkg
Pkg.add("Reactant")

using DifferentiationInterface
using DifferentiationInterfaceTest
using Reactant

backend = AutoReactant()

test_differentiation(
    backend, DifferentiationInterfaceTest.default_scenarios();
    excluded = vcat(SECOND_ORDER, :jacobian, :derivative, :pushforward, :pullback),
    logging = true
)
