using Pkg
Pkg.add(["FiniteDifferences", "Enzyme", "Flux", "Zygote"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using Random
using Zygote: Zygote
using Test

LOGGING = get(ENV, "CI", "false") == "false"

test_differentiation(
    [
        AutoZygote(),
        # AutoEnzyme(), # TODO a few scenarios fail
    ],
    DIT.flux_scenarios(Random.MersenneTwister(0));
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-4,
    scenario_intact=false,  # TODO: why?
    logging=LOGGING,
)
