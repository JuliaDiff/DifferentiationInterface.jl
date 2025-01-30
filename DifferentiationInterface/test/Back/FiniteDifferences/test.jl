using Pkg
Pkg.add("FiniteDifferences")

using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDifferences: FiniteDifferences
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

test_differentiation(
    AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)),
    default_scenarios(; include_constantified=true, include_cachified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
