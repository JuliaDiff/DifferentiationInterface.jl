using Pkg
Pkg.add("Enzyme")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using StaticArrays
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

function remove_matrix_inputs(scens::Vector{<:Scenario})  # TODO: remove
    if VERSION < v"1.11"
        return scens
    else
        # for https://github.com/EnzymeAD/Enzyme.jl/issues/2071
        return filter(s -> s.x isa Union{Number,AbstractVector}, scens)
    end
end

backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
    AutoEnzyme(; mode=nothing, function_annotation=Enzyme.Const),
]

duplicated_backends = [
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Duplicated),
]

@testset "First order" begin
    test_differentiation(
        backends, default_scenarios(); excluded=SECOND_ORDER, logging=LOGGING
    )

    test_differentiation(
        backends[1:3],
        default_scenarios(; include_normal=false, include_constantified=true);
        excluded=SECOND_ORDER,
        logging=LOGGING,
    )

    test_differentiation(
        backends[2:3],
        default_scenarios(; include_normal=false, include_cachified=true);
        excluded=SECOND_ORDER,
        logging=LOGGING,
    )

    test_differentiation(
        duplicated_backends,
        default_scenarios(; include_normal=false, include_closurified=true);
        excluded=SECOND_ORDER,
        logging=LOGGING,
    )
end

test_differentiation(
    AutoEnzyme(mode=Enzyme.Reverse),
    default_scenarios(; include_normal=false, include_cachified=true);
    excluded=vcat(SECOND_ORDER, :jacobian, :gradient, :pushforward, :derivative),
    logging=LOGGING,
)

#=
# TODO: reactivate type stability tests

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward),  # TODO: add more
    default_scenarios(; include_batchified=false);
    correctness=false,
    type_stability=:prepared,
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
=#

@testset "Second order" begin
    test_differentiation(
        [
            AutoEnzyme(),
            SecondOrder(
                AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Forward)
            ),
        ],
        remove_matrix_inputs(default_scenarios(; include_constantified=true));
        excluded=FIRST_ORDER,
        logging=LOGGING,
    )

    test_differentiation(
        AutoEnzyme(; mode=Enzyme.Forward);
        excluded=vcat(FIRST_ORDER, [:hessian, :hvp]),
        logging=LOGGING,
    )
end

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse.(AutoEnzyme(; function_annotation=Enzyme.Const)),
        if VERSION < v"1.11"
            sparse_scenarios()
        else
            filter(sparse_scenarios()) do s
                # for https://github.com/EnzymeAD/Enzyme.jl/issues/2168
                (s.x isa AbstractVector) &&
                    (s.f != DIT.sumdiffcube) &&
                    (s.f != DIT.sumdiffcube_mat)
            end
        end;
        sparsity=true,
        logging=LOGGING,
    )
end

@testset "Static" begin
    filtered_static_scenarios = filter(static_scenarios()) do s
        DIT.operator_place(s) == :out && DIT.function_place(s) == :out
    end

    test_differentiation(
        [AutoEnzyme(; mode=Enzyme.Forward), AutoEnzyme(; mode=Enzyme.Reverse)],
        filtered_static_scenarios;
        excluded=SECOND_ORDER,
        logging=LOGGING,
    )
end

@testset "Wrong type tangent" begin
    x, y = [1.0], [1.0]
    # pushforward
    dx, dy = [2.0f0], [0.0f0]
    @test pushforward!(
        copy, (similar(dy),), AutoEnzyme(; mode=Enzyme.Forward), x, (dx,)
    )[1] == [2]
    @test pushforward!(
        copy, (similar(dy), similar(dy)), AutoEnzyme(; mode=Enzyme.Forward), x, (dx, -dx)
    )[2] == -[2]
    @test pushforward!(
        copyto!, y, (similar(dy),), AutoEnzyme(; mode=Enzyme.Forward), x, (dx,)
    )[1] == [2]
    @test pushforward!(
        copyto!,
        y,
        (similar(dy), similar(dy)),
        AutoEnzyme(; mode=Enzyme.Forward),
        x,
        (dx, -dx),
    )[2] == -[2]
    # pullback
    dy, dx = [2.0f0], [0.0f0]
    @test pullback!(copy, (similar(dx),), AutoEnzyme(; mode=Enzyme.Reverse), x, (dy,))[1] ==
        [2]
    @test pullback!(
        copy, (similar(dx), similar(dx)), AutoEnzyme(; mode=Enzyme.Reverse), x, (dy, -dy)
    )[2] == -[2]
    @test pullback!(
        copyto!, y, (similar(dx),), AutoEnzyme(; mode=Enzyme.Reverse), x, (dy,)
    )[1] == [2]
    @test pullback!(
        copyto!,
        y,
        (similar(dx), similar(dx)),
        AutoEnzyme(; mode=Enzyme.Reverse),
        x,
        (dy, -dy),
    )[2] == -[2]
end
