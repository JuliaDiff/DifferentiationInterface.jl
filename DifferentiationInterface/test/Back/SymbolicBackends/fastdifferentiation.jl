using Pkg
Pkg.add("FastDifferentiation")

using DifferentiationInterface, DifferentiationInterfaceTest
using SparseMatrixColorings
using LinearAlgebra
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
    default_scenarios(;
        include_constantified=true, include_cachified=true, use_tuples=false
    );
    logging=LOGGING,
);

test_differentiation(
    AutoSparse(AutoFastDifferentiation()),
    sparse_scenarios(; band_sizes=0:-1);
    sparsity=true,
    logging=LOGGING,
);

@testset "SparseMatrixColorings access" begin
    x = rand(10)
    backend = AutoSparse(AutoFastDifferentiation())
    jac_prep = prepare_jacobian(copy, backend, x)
    jac!_prep = prepare_jacobian(copyto!, similar(x), backend, x)
    hess_prep = prepare_hessian(x -> sum(abs2, x), backend, x)
    @test sparsity_pattern(jac_prep) == Diagonal(trues(10))
    @test sparsity_pattern(jac!_prep) == Diagonal(trues(10))
    @test sparsity_pattern(hess_prep) == Diagonal(trues(10))
end
