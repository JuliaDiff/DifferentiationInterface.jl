using Pkg
Pkg.add("FiniteDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
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
    default_scenarios(; include_constantified=true, include_cachified=true);
    excluded=[:second_derivative, :hvp],
    logging=LOGGING,
);

@testset verbose = true "Complex number support" begin
    backend = AutoSparse(AutoFiniteDiff(); coloring_algorithm=GreedyColoringAlgorithm())
    x = float.(1:3) .+ im
    @test_nowarn jacobian(identity, backend, x)
    @test_nowarn jacobian(copyto!, similar(x), backend, x)
    @test_nowarn hessian(sum, backend, x)
end
