using Pkg
Pkg.add("Mooncake")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())]

test_differentiation(
    backends,
    default_scenarios(; include_constantified=true, include_cachified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
