using Pkg
Pkg.add(url = "https://github.com/EnzymeAD/Enzyme.jl")
Pkg.add("Reactant")

using DifferentiationInterface
using DifferentiationInterfaceTest
using Reactant
using Test

backend = AutoReactant()

@test check_available(backend)
@test check_inplace(backend)

test_differentiation(
    backend, DifferentiationInterfaceTest.default_scenarios(;
        include_constantified = true, include_cachified = false
    );
    excluded = vcat(SECOND_ORDER, :jacobian, :derivative, :pushforward, :pullback),
    logging = false
)
