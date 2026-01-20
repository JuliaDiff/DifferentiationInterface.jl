include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ExplicitImports
using HyperHessians
using Test

check_no_implicit_imports(DifferentiationInterface)

backends = [
    DI.AutoHyperHessians(),
    DI.AutoHyperHessians(; chunksize = 4),
]

for backend in backends
    @test DI.check_available(backend)
    @test DI.check_inplace(backend)
end

scenarios = default_scenarios(; include_constantified = true, include_cachified = true)

test_differentiation(
    backends, scenarios;
    excluded = FIRST_ORDER, logging = LOGGING,
)

test_differentiation(
    DI.AutoHyperHessians(), scenarios;
    correctness = false,
    type_stability = safetypestab(:prepared),
    excluded = FIRST_ORDER,
    logging = LOGGING,
)
