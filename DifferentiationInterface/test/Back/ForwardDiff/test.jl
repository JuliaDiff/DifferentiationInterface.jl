using Pkg
Pkg.add("ForwardDiff")

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer, SparseMatrixColorings
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

dense_backends = [AutoForwardDiff(; tag=:hello), AutoForwardDiff(; chunksize=5)]

sparse_backends =
    AutoSparse.(
        dense_backends;
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense backends

test_differentiation(
    dense_backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(); correctness=false, type_stability=true, logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(; chunksize=5);
    correctness=false,
    type_stability=(; preparation=true, prepared_op=true, unprepared_op=false),
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

test_differentiation(
    sparse_backends,
    sparse_scenarios(; include_constantified=true);
    sparsity=true,
    logging=LOGGING,
);
