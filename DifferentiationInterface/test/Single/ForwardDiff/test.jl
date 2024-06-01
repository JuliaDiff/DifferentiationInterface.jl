using DifferentiationInterface, DifferentiationInterfaceTest
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

test_differentiation(dense_backends; logging=LOGGING);

test_differentiation(
    dense_backends;
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
    excluded=[JacobianScenario],
    second_order=false,
    logging=LOGGING,
);

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[
        DerivativeScenario,
        GradientScenario,
        HVPScenario,
        PullbackScenario,
        PushforwardScenario,
        SecondDerivativeScenario,
    ],
    logging=LOGGING,
);

test_differentiation(sparse_backends, sparse_scenarios(); sparsity=true, logging=LOGGING);
