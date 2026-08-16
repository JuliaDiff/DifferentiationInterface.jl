using DifferentiationInterface
using DifferentiationInterfaceTest
using Aqua: Aqua
using Base: get_extension
using ExplicitImports
using JET: JET
using SparseMatrixColorings: SparseMatrixColorings
using Test
import Chairmarks

const DIT = DifferentiationInterfaceTest

if isdefined(JET, :ReportMatcher)
    # on Julia 1.12, a type instability inside `Test.@testset` itself (not in our code)
    # makes JET flag `test_differentiation`'s own outer testset as a possible error, see
    # https://github.com/JuliaLang/julia/issues/59316 (fixed on 1.13, unreleased on 1.12).
    # Ignore just that report, matched by the mangled name JET reports the keyword-body
    # method under. Older JET releases (resolved on Julia < 1.12, due to their own upper
    # Julia compat bounds) predate `JET.ReportMatcher` and don't need this: the
    # underlying Julia bug only affects 1.12.
    struct IgnoreTestsetInstability <: JET.ReportMatcher
        name::Symbol
    end
    function JET.match_report(
            matcher::IgnoreTestsetInstability, @nospecialize(report::JET.InferenceErrorReport)
        )
        report isa JET.UndefVarErrorReport || return false
        def = last(report.vst).linfo.def
        return def isa Method && startswith(String(def.name), "#$(matcher.name)#")
    end
end

@testset "Aqua" begin
    Aqua.test_all(DifferentiationInterfaceTest; ambiguities = false, undocumented_names = true)
end
@testset verbose = true "JET" begin
    extra_jetconfigs = isdefined(JET, :ReportMatcher) ?
        (; ignored_modules = (IgnoreTestsetInstability(:test_differentiation),)) : (;)
    JET.test_package(
        DIT;
        extra_jetconfigs...,
        target_modules = (
            DIT,
            filter(
                !isnothing, (
                    get_extension(DIT, :DifferentiationInterfaceTestChairmarksExt),
                    get_extension(DIT, :DifferentiationInterfaceTestComponentArraysExt),
                    get_extension(DIT, :DifferentiationInterfaceTestJETExt),
                    get_extension(DIT, :DifferentiationInterfaceTestJLArraysExt),
                    get_extension(DIT, :DifferentiationInterfaceTestStaticArraysExt),
                )
            )...,
        )
    )
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
