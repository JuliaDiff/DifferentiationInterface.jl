using ADTypes
using ComponentArrays: ComponentArrays
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ForwardDiff: ForwardDiff
using JLArrays: JLArrays
using Random
using SparseConnectivityTracer
using SparseMatrixColorings
using StaticArrays: StaticArrays
using Zygote: Zygote

## Generate all scenarios

gpu_scenarios(;
    include_constantified = true,
    include_closurified = true,
    include_batchified = true,
    include_cachified = true,
    use_tuples = true,
)
static_scenarios(;
    include_constantified = true,
    include_closurified = true,
    include_batchified = true,
    include_cachified = true,
    use_tuples = true,
)

## Weird arrays

test_differentiation(
    AutoForwardDiff(),
    DIT.no_matrices(static_scenarios());
    benchmark = :prepared,
    logging = LOGGING,
)

test_differentiation(AutoForwardDiff(), component_scenarios(); logging = LOGGING)

test_differentiation(AutoZygote(), gpu_scenarios(); excluded = SECOND_ORDER, logging = LOGGING)

## Closures & caches

test_differentiation(
    AutoFiniteDiff(),
    default_scenarios(;
        include_normal = false,
        include_closurified = true,
        include_cachified = true,
        use_tuples = true,
    );
    excluded = SECOND_ORDER,
    logging = LOGGING,
);
