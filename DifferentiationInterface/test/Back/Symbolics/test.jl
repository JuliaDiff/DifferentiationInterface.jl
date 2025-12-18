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

@testset "Non-numeric arguments" begin
    function differentiate_me!(out, x, c)
        @assert c isa Any # Just to use it somewhere
        return copyto!(out, x)
    end
    function differentiate_me(x, c)
        tmp = similar(x)
        differentiate_me!(tmp, x, c)
        return tmp
    end
    x = rand(10)
    xbuffer = copy(x)
    c = "I am a string"
    backend = AutoSymbolics()
    jac_prep = prepare_jacobian(differentiate_me, backend, x, Constant(c))
    jac!_prep = prepare_jacobian(differentiate_me!, xbuffer, backend, x, Constant(c))
    @test jacobian(differentiate_me, jac_prep, backend, x, Constant(c)) ≈ I
    @test jacobian(differentiate_me!, xbuffer, jac!_prep, backend, x, Constant(c)) ≈ I
end
