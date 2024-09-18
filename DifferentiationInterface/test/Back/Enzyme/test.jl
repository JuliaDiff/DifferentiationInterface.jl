using Pkg
Pkg.add("Enzyme")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using SparseConnectivityTracer, SparseMatrixColorings
using StableRNGs
using Test

LOGGING = get(ENV, "CI", "false") == "false"

dense_backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=nothing, function_annotation=Enzyme.Const),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const),
    AutoEnzyme(; mode=Enzyme.Reverse),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
]

nested_dense_backends = [
    DifferentiationInterface.nested(AutoEnzyme(; mode=Enzyme.Forward)),
    DifferentiationInterface.nested(AutoEnzyme(; mode=Enzyme.Reverse)),
]

duplicated_function_backends = [
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Duplicated),
]

sparse_backends =
    AutoSparse.(
        dense_backends,
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

@testset "Checks" begin
    @testset "Check $(typeof(backend))" for backend in vcat(dense_backends, sparse_backends)
        @test check_available(backend)
        @test check_inplace(backend)
    end
end

## Dense backends

test_differentiation(
    vcat(dense_backends, nested_dense_backends),
    default_scenarios();
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    duplicated_function_backends,
    default_scenarios(; include_normal=false, include_closurified=true);
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    [
        AutoEnzyme(; mode=nothing),
        AutoEnzyme(; mode=Enzyme.Reverse),
        SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Reverse)),
        SecondOrder(AutoEnzyme(; mode=Enzyme.Forward), AutoEnzyme(; mode=Enzyme.Reverse)),
    ];
    first_order=false,
    excluded=[:second_derivative],
    logging=LOGGING,
);

test_differentiation(
    [AutoEnzyme(; mode=nothing), AutoEnzyme(; mode=Enzyme.Forward)];
    first_order=false,
    excluded=[:hessian, :hvp],
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward),  # TODO: add more
    default_scenarios(; include_batchified=false);
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[:derivative, :gradient, :pullback, :pushforward],
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    sparse_backends, sparse_scenarios(); second_order=false, sparsity=true, logging=LOGGING
);
