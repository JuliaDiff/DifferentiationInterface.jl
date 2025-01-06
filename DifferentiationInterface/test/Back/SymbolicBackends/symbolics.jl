using Pkg
Pkg.add("Symbolics")

using DifferentiationInterface, DifferentiationInterfaceTest
using Symbolics: Symbolics
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoSymbolics(), AutoSparse(AutoSymbolics())]
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(AutoSymbolics(); logging=LOGGING);

test_differentiation(
    AutoSparse(AutoSymbolics()),
    sparse_scenarios(; band_sizes=0:-1);
    sparsity=true,
    logging=LOGGING,
);
