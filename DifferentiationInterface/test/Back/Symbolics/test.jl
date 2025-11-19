using Pkg
Pkg.activate(@__DIR__)
include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using SparseMatrixColorings
using LinearAlgebra
using Symbolics: Symbolics
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

for backend in [AutoSymbolics(), AutoSparse(AutoSymbolics())]
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    AutoSymbolics(), default_scenarios(; include_constantified = true); logging = LOGGING
);

test_differentiation(
    AutoSymbolics(),
    default_scenarios(; include_normal = false, include_cachified = true, use_tuples = false);
    logging = LOGGING,
);

test_differentiation(
    AutoSparse(AutoSymbolics()),
    sparse_scenarios(; band_sizes = 0:-1);
    sparsity = true,
    logging = LOGGING,
);

@testset "SparseMatrixColorings access" begin
    x = rand(10)
    backend = AutoSparse(AutoSymbolics())
    jac_prep = prepare_jacobian(copy, backend, x)
    jac!_prep = prepare_jacobian(copyto!, similar(x), backend, x)
    hess_prep = prepare_hessian(x -> sum(abs2, x), backend, x)
    @test sparsity_pattern(jac_prep) == Diagonal(trues(10))
    @test sparsity_pattern(jac!_prep) == Diagonal(trues(10))
    @test sparsity_pattern(hess_prep) == Diagonal(trues(10))
end
