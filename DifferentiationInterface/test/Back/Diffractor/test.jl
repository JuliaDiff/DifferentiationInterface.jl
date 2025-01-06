using Pkg
Pkg.add("Diffractor")

using DifferentiationInterface, DifferentiationInterfaceTest
using Diffractor: Diffractor
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoDiffractor()]
    @test check_available(backend)
    @test !check_inplace(backend)
end

test_differentiation(
    AutoDiffractor(),
    default_scenarios(; linalg=false);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
