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

excluded_ops = [:pushforward, :pullback, :jacobian, :derivative, :gradient]

scenarios = default_scenarios(; include_constantified = true)

test_differentiation(
    backends, scenarios;
    excluded = excluded_ops, logging = LOGGING,
)

test_differentiation(
    DI.AutoHyperHessians(), scenarios;
    correctness = false,
    type_stability = safetypestab(:prepared),
    excluded = excluded_ops,
    logging = LOGGING,
)
