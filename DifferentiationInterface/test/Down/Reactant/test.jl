using Pkg
Pkg.add("Enzyme")
Pkg.add(; url="https://github.com/EnzymeAD/Reactant.jl")

using DifferentiationInterface
using DifferentiationInterface: ReactantBackend
using DifferentiationInterfaceTest
using Enzyme: Enzyme
using LinearAlgebra
using Reactant: Reactant
using Test

@assert !isnothing(
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReactantExt)
)

LOGGING = get(ENV, "CI", "false") == "false"

scenarios = [
    Scenario{:gradient,:out}(sum, [1.0, 2.0]; res1=ones(2)),
    Scenario{:gradient,:in}(sum, [1.0, 2.0]; res1=ones(2)),
]

test_differentiation(ReactantBackend(AutoEnzyme()), scenarios; logging=LOGGING)
