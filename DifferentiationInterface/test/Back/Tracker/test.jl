using Pkg
Pkg.add("Tracker")

using DifferentiationInterface, DifferentiationInterfaceTest
using Tracker: Tracker
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

test_differentiation(
    AutoTracker(),
    default_scenarios(; include_constantified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
