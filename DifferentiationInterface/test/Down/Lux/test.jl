using Pkg
Pkg.add(["ForwardDiff", "Lux", "LuxTestUtils", "Zygote"])

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using Lux: Lux
using LuxTestUtils: LuxTestUtils
using Random

LOGGING = get(ENV, "CI", "false") == "false"

test_differentiation(
    AutoZygote(),
    DIT.lux_scenarios(Random.Xoshiro(63));
    isapprox = DIT.lux_isapprox,
    rtol = 1.0f-2,
    atol = 1.0f-3,
    scenario_intact = false,  # TODO: why?
    logging = LOGGING,
)
