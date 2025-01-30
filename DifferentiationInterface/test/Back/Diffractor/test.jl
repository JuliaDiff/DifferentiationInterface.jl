using Pkg
Pkg.add("Diffractor")

using DifferentiationInterface, DifferentiationInterfaceTest
using Diffractor: Diffractor
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

test_differentiation(
    AutoDiffractor(),
    default_scenarios(; linalg=false);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
