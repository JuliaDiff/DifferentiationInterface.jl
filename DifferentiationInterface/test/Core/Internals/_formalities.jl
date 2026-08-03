# tested first so that the suite fails quickly

using Aqua: Aqua
using Base: get_extension
using DifferentiationInterface
using ExplicitImports
using JET: JET
using Test
using SparseMatrixColorings
using SparseArrays

const DI = DifferentiationInterface

@testset "Aqua" begin
    Aqua.test_all(DifferentiationInterface; ambiguities = false, undocumented_names = true)
end

@testset "JET" begin
    JET.test_package(
        DI;
        target_modules = (
            DI,
            get_extension(DI, :DifferentiationInterfaceChainRulesCoreExt),
            # get_extension(DI, :DifferentiationInterfaceDiffractorExt),
            get_extension(DI, :DifferentiationInterfaceEnzymeExt),
            get_extension(DI, :DifferentiationInterfaceFastDifferentiationExt),
            get_extension(DI, :DifferentiationInterfaceFiniteDiffExt),
            get_extension(DI, :DifferentiationInterfaceFiniteDifferencesExt),
            get_extension(DI, :DifferentiationInterfaceForwardDiffExt),
            get_extension(DI, :DifferentiationInterfaceGPUArraysCoreExt),
            get_extension(DI, :DifferentiationInterfaceHyperHessiansExt),
            get_extension(DI, :DifferentiationInterfaceGTPSAExt),
            get_extension(DI, :DifferentiationInterfaceMooncakeExt),
            get_extension(DI, :DifferentiationInterfacePolyesterForwardDiffExt),
            get_extension(DI, :DifferentiationInterfaceReverseDiffExt),
            get_extension(DI, :DifferentiationInterfaceSparseArraysExt),
            get_extension(DI, :DifferentiationInterfaceSparseConnectivityTracerExt),
            get_extension(DI, :DifferentiationInterfaceSparseMatrixColoringsExt),
            get_extension(DI, :DifferentiationInterfaceStaticArraysExt),
            get_extension(DI, :DifferentiationInterfaceSymbolicsExt),
            get_extension(DI, :DifferentiationInterfaceTrackerExt),
            get_extension(DI, :DifferentiationInterfaceZygoteExt),
        )
    )
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
        @test check_all_explicit_imports_are_public(DifferentiationInterface) === nothing
        @test_skip check_all_qualified_accesses_are_public(DifferentiationInterface) ===
            nothing
    end
end
