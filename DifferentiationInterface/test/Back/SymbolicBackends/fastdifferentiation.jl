using Pkg
Pkg.add("FastDifferentiation")

using DifferentiationInterface, DifferentiationInterfaceTest
using FastDifferentiation: FastDifferentiation
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoFastDifferentiation(), AutoSparse(AutoFastDifferentiation())]
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    AutoFastDifferentiation(),
    filter(
        default_scenarios() do s
            !(s.x isa Matrix) && !(s.y isa Matrix)
        end,
    );
    logging=LOGGING,
);

test_differentiation(
    AutoSparse(AutoFastDifferentiation()),
    sparse_scenarios(; band_sizes=0:-1);
    sparsity=true,
    logging=LOGGING,
);
