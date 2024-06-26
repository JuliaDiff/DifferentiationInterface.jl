using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterfaceTest: add_batchified!
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer, SparseMatrixColorings
using Test

dense_backends = [AutoForwardDiff(), AutoForwardDiff(; chunksize=2, tag=:hello)]

sparse_backends = [
    AutoSparse(
        AutoForwardDiff();
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

## Dense backends

test_differentiation(dense_backends, add_batchified!(default_scenarios()); logging=LOGGING);

test_differentiation(
    fromprimitive_backends,
    add_batchified!(default_scenarios());
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    dense_backends,
    # ForwardDiff accesses individual indices
    vcat(component_scenarios(), static_scenarios());
    # jacobian is super slow for some reason
    excluded=[:jacobian],
    second_order=false,
    logging=LOGGING,
);

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[:derivative, :gradient, :hvp, :pullback, :pushforward, :second_derivative],
    logging=LOGGING,
);

test_differentiation(sparse_backends, sparse_scenarios(); sparsity=true, logging=LOGGING);
