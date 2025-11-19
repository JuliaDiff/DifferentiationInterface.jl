include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using Tracker: Tracker
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

for backend in [AutoTracker()]
    @test check_available(backend)
    @test !check_inplace(backend)
end

test_differentiation(
    AutoTracker(),
    default_scenarios(; include_constantified = true);
    excluded = SECOND_ORDER,
    logging = LOGGING,
);
