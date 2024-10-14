using DifferentiationInterface
using DifferentiationInterfaceTest
using Aqua: Aqua
using ExplicitImports
using JET: JET
using JuliaFormatter: JuliaFormatter
using SparseMatrixColorings: SparseMatrixColorings
using Test

@testset "Aqua" begin
    Aqua.test_all(
        DifferentiationInterfaceTest; ambiguities=false, deps_compat=(check_extras = false)
    )
end
@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(
        DifferentiationInterfaceTest; verbose=false, overwrite=false
    )
end
@testset verbose = true "JET" begin
    JET.test_package(DifferentiationInterfaceTest; target_defined_modules=true)
end

@testset "ExplicitImports" begin
    @test_broken check_no_implicit_imports(DifferentiationInterfaceTest) === nothing
    @test_broken check_no_stale_explicit_imports(DifferentiationInterfaceTest) === nothing
    @test_broken check_all_explicit_imports_via_owners(DifferentiationInterfaceTest) ===
        nothing
    @test_broken check_all_explicit_imports_are_public(DifferentiationInterfaceTest) ===
        nothing
    @test check_all_qualified_accesses_via_owners(DifferentiationInterfaceTest) === nothing
    @test_broken check_all_qualified_accesses_are_public(DifferentiationInterfaceTest) ===
        nothing
    @test check_no_self_qualified_accesses(DifferentiationInterfaceTest) === nothing
end