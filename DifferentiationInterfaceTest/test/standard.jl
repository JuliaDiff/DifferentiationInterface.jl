using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Random

LOGGING = get(ENV, "CI", "false") == "false"

## Dense

test_differentiation(
    AutoForwardDiff(), default_scenarios(; include_constantified=false); logging=LOGGING
)

## Complex

test_differentiation(
    AutoFiniteDiff(), vcat(complex_scenarios(), complex_sparse_scenarios()); logging=LOGGING
)

## Sparse

sparse_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

test_differentiation(
    sparse_backend,
    sparse_scenarios(; include_constantified=true);
    sparsity=true,
    logging=LOGGING,
)
