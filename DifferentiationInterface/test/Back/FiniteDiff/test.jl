using Pkg
Pkg.add(; url="https://github.com/JuliaDiff/FiniteDiff.jl", rev="os/finite_difference_jvp`")

using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: DenseSparsityDetector
using FiniteDiff: FiniteDiff
using SparseMatrixColorings
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoFiniteDiff()]
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    AutoFiniteDiff(),
    default_scenarios();
    excluded=[
        :second_derivative, :hvp, :hessian, :jacobian, :derivative, :gradient, :pullback
    ],
    logging=LOGGING,
);

test_differentiation(
    AutoFiniteDiff(),
    default_scenarios(; include_constantified=true, include_cachified=true);
    excluded=[:second_derivative, :hvp],
    logging=LOGGING,
);

test_differentiation(
    [
        AutoFiniteDiff(; relstep=cbrt(eps(Float64))),
        AutoFiniteDiff(; relstep=cbrt(eps(Float64)), absstep=cbrt(eps(Float64))),
    ];
    excluded=[:second_derivative, :hvp],
    logging=LOGGING,
);

@testset "Complex" begin
    test_differentiation(AutoFiniteDiff(), complex_scenarios(); logging=LOGGING)
    test_differentiation(
        AutoSparse(
            AutoFiniteDiff();
            sparsity_detector=DenseSparsityDetector(AutoFiniteDiff(); atol=1e-5),
            coloring_algorithm=GreedyColoringAlgorithm(),
        ),
        complex_sparse_scenarios();
        logging=LOGGING,
    )
end;
