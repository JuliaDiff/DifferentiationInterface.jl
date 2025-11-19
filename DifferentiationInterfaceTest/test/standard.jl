using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Random

## Dense

test_differentiation(
    [AutoForwardDiff(), AutoForwardDiff(; chunksize = 100)],
    default_scenarios(; include_smaller = true, include_constantified = true);
    logging = LOGGING,
)

test_differentiation(
    [AutoForwardDiff(), AutoFiniteDiff(; relstep = 1.0e-5)],
    default_scenarios(;
        include_batchified = false,
        include_normal = false,
        include_cachified = true,
        include_constantorcachified = true,
    );
    logging = LOGGING,
)

test_differentiation(
    [AutoForwardDiff()], empty_scenarios(); excluded = [:gradient], logging = LOGGING
)
test_differentiation(
    [AutoFiniteDiff()], empty_scenarios(); excluded = [:jacobian], logging = LOGGING
)

## Sparse

sparse_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

test_differentiation(
    sparse_backend,
    sparse_scenarios(; include_cachified = true, use_tuples = false);
    sparsity = true,
    logging = LOGGING,
)

## Complex

test_differentiation(
    AutoFiniteDiff(), vcat(complex_scenarios(), complex_sparse_scenarios()); logging = LOGGING
)
