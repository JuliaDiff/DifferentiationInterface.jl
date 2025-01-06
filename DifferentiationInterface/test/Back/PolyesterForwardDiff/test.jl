using Pkg
Pkg.add("PolyesterForwardDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
using PolyesterForwardDiff: PolyesterForwardDiff
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

backends = [
    AutoPolyesterForwardDiff(; tag=:hello),  #
    AutoPolyesterForwardDiff(; chunksize=2),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    backends, default_scenarios(; include_constantified=true); logging=LOGGING
);
