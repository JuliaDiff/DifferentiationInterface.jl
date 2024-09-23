using Pkg
Pkg.add(["FiniteDiff"])
# Pkg.add(["FiniteDiff", "Lux", "LuxTestUtils"])

using ADTypes
using ComponentArrays: ComponentArrays
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using ForwardDiff: ForwardDiff
using JLArrays: JLArrays
# using Lux: Lux
# using LuxTestUtils: LuxTestUtils
using Random
using SparseConnectivityTracer
using SparseMatrixColorings
using StaticArrays: StaticArrays
using Zygote: Zygote

LOGGING = get(ENV, "CI", "false") == "false"

## Weird arrays

test_differentiation(
    AutoForwardDiff(), vcat(component_scenarios(), static_scenarios()); logging=LOGGING
)

test_differentiation(AutoZygote(), gpu_scenarios(); second_order=false, logging=LOGGING)

## Closures

test_differentiation(
    AutoFiniteDiff(),
    default_scenarios(; include_normal=false, include_closurified=true);
    second_order=false,
    logging=LOGGING,
);

## Neural nets

Random.seed!(0)

test_differentiation(
    AutoZygote(),
    DIT.flux_scenarios();
    isapprox=DIT.flux_isapprox,
    rtol=1e-2,
    atol=1e-6,
    scenario_intact=false,
    logging=LOGGING,
)

#=
test_differentiation(
    AutoZygote(),
    DIT.lux_scenarios(Random.Xoshiro(63));
    isapprox=DIT.lux_isapprox,
    rtol=1.0f-2,
    atol=1.0f-3,
    scenario_intact=false,
    logging=LOGGING,
)
=#
