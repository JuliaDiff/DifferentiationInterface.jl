# tested first so that the suite fails quickly

using Aqua: Aqua
using DifferentiationInterface
using ExplicitImports
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test
using SparseMatrixColorings
using SparseArrays

@testset "Aqua" begin
    Aqua.test_all(DifferentiationInterface; ambiguities=false, undocumented_names=true)
end

@testset "JET" begin
    JET.test_package(DifferentiationInterface; target_defined_modules=true)
end

@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(DifferentiationInterface; verbose=false, overwrite=false)
end

@testset "Documentation" begin
    if VERSION >= v"1.11"
        @test isempty(Docs.undocumented_names(DifferentiationInterface))
    end
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DifferentiationInterface) === nothing
    @test check_no_stale_explicit_imports(DifferentiationInterface) === nothing
    @test check_all_explicit_imports_via_owners(DifferentiationInterface) === nothing
    @test check_all_qualified_accesses_via_owners(DifferentiationInterface) === nothing
    @test check_no_self_qualified_accesses(DifferentiationInterface) === nothing
    if VERSION >= v"1.11"
        @test check_all_explicit_imports_are_public(DifferentiationInterface;) === nothing
        @test_skip check_all_qualified_accesses_are_public(DifferentiationInterface) ===
            nothing
    end
end
