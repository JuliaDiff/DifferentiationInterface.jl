# see https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/855

using Pkg
Pkg.activate(@__DIR__)
include("../../testutils.jl")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using LinearAlgebra
using StaticArrays
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

backends = [
    AutoEnzyme(; mode = nothing),
    AutoEnzyme(; mode = Enzyme.Forward),
    AutoEnzyme(; mode = Enzyme.Reverse),
    AutoEnzyme(; mode = nothing, function_annotation = Enzyme.Const),
]

duplicated_backends = [
    AutoEnzyme(; mode = Enzyme.Forward, function_annotation = Enzyme.Duplicated),
    AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated),
]

@testset "Checks" begin
    @testset "Check $(typeof(backend))" for backend in backends
        @test check_available(backend)
        @test check_inplace(backend)
    end
end;

@testset "First order" begin
    test_differentiation(
        backends, default_scenarios(); excluded = SECOND_ORDER, logging = LOGGING
    )

    test_differentiation(
        backends[1:3],
        default_scenarios(; include_normal = false, include_constantified = true);
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )

    test_differentiation(
        backends[2:3],
        default_scenarios(;
            include_normal = false,
            include_cachified = true,
            include_constantorcachified = true,
            use_tuples = true,
        );
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )

    test_differentiation(
        duplicated_backends,
        default_scenarios(; include_normal = false, include_closurified = true);
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )
end

@testset "Second order" begin
    test_differentiation(
        [
            AutoEnzyme(),
            SecondOrder(
                AutoEnzyme(; mode = Enzyme.Reverse), AutoEnzyme(; mode = Enzyme.Forward)
            ),
        ],
        default_scenarios(; include_constantified = true, include_cachified = true);
        excluded = FIRST_ORDER,
        logging = LOGGING,
    )
end

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse.(AutoEnzyme(; function_annotation = Enzyme.Const)),
        if VERSION < v"1.11"
            sparse_scenarios()
        else
            filter(s -> s.x isa AbstractVector, sparse_scenarios())
        end;
        sparsity = true,
        logging = LOGGING,
    )
end

@testset "Static" begin
    filtered_static_scenarios = filter(static_scenarios()) do s
        DIT.operator_place(s) == :out && DIT.function_place(s) == :out
    end

    test_differentiation(
        [AutoEnzyme(; mode = Enzyme.Forward), AutoEnzyme(; mode = Enzyme.Reverse)],
        filtered_static_scenarios;
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )
end

@testset "Coverage" begin
    # ConstantOrCache without cache
    f_nocontext(x, p) = x
    @test I == DifferentiationInterface.jacobian(
        f_nocontext, AutoEnzyme(; mode = Enzyme.Forward), rand(10), ConstantOrCache(nothing)
    )
    @test I == DifferentiationInterface.jacobian(
        f_nocontext, AutoEnzyme(; mode = Enzyme.Reverse), rand(10), ConstantOrCache(nothing)
    )
end

@testset "Hints" begin
    @testset "MutabilityError" begin
        f = let
            cache = [0.0]
            x -> sum(copyto!(cache, x))
        end

        e = nothing
        try
            gradient(f, AutoEnzyme(), [1.0])
        catch e
        end
        msg = sprint(showerror, e)
        @test occursin("AutoEnzyme", msg)
        @test occursin("function_annotation", msg)
        @test occursin("ADTypes", msg)
    end

    @testset "RuntimeActivityError" begin
        function g(active_var, constant_var, cond)
            if cond
                return active_var
            else
                return constant_var
            end
        end

        function h(active_var, constant_var, cond)
            return [g(active_var, constant_var, cond), g(active_var, constant_var, cond)]
        end

        e = nothing
        try
            pushforward(
                h,
                AutoEnzyme(; mode = Enzyme.Forward),
                [1.0],
                ([1.0],),
                Constant([1.0]),
                Constant(true),
            )
        catch e
        end
        msg = sprint(showerror, e)
        @test_broken occursin("AutoEnzyme", msg)
        @test_broken occursin("ADTypes", msg)
    end
end

@testset "Empty arrays" begin
    test_differentiation(
        [AutoEnzyme(; mode = Enzyme.Forward), AutoEnzyme(; mode = Enzyme.Reverse)],
        empty_scenarios();
        excluded = [:jacobian],
    )
end;
