using Pkg
Pkg.add("Enzyme")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using SparseConnectivityTracer, SparseMatrixColorings
using StableRNGs
using Test

dense_backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
]

nested_dense_backends = [
    DifferentiationInterface.nested(AutoEnzyme(; mode=Enzyme.Forward)),
    DifferentiationInterface.nested(AutoEnzyme(; mode=Enzyme.Reverse)),
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
        @test check_twoarg(backend)
        @test check_hessian(backend; verbose=false)
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
    AutoEnzyme(; mode=Enzyme.Forward);  # TODO: add more
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
