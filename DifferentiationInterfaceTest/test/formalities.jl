using DifferentiationInterface
using DifferentiationInterfaceTest
using Aqua: Aqua
using ExplicitImports
using JET: JET
using JuliaFormatter: JuliaFormatter
using SparseMatrixColorings: SparseMatrixColorings
using Test

@testset "Aqua" begin
    Aqua.test_all(DifferentiationInterfaceTest; ambiguities=false, undocumented_names=true)
end
@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(
        DifferentiationInterfaceTest; verbose=false, overwrite=false
    )
end
@testset verbose = true "JET" begin
    JET.test_package(DifferentiationInterfaceTest; target_defined_modules=true)
end

@testset "Documentation" begin
    if VERSION >= v"1.11"
        @test isempty(Docs.undocumented_names(DifferentiationInterfaceTest))
    end
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DifferentiationInterfaceTest) === nothing
    @test_broken check_no_stale_explicit_imports(DifferentiationInterfaceTest) === nothing
    @test_broken check_all_explicit_imports_via_owners(DifferentiationInterfaceTest) ===
        nothing
    @test check_all_qualified_accesses_via_owners(DifferentiationInterfaceTest) === nothing
    @test check_no_self_qualified_accesses(DifferentiationInterfaceTest) === nothing
    if VERSION >= v"1.11"
        @test_broken check_all_explicit_imports_are_public(DifferentiationInterfaceTest) ===
            nothing
        @test_broken check_all_qualified_accesses_are_public(
            DifferentiationInterfaceTest
        ) === nothing
    end
end
