using Pkg
Pkg.add("FiniteDifferences")

using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDifferences: FiniteDifferences
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1))]
    @test check_available(backend)
    @test !check_inplace(backend)
    @test DifferentiationInterface.inner_preparation_behavior(backend) isa
        DifferentiationInterface.PrepareInnerSimple
end

test_differentiation(
    AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)),
    default_scenarios(; include_constantified=true, include_cachified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
