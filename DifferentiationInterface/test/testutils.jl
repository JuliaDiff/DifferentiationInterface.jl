using ADTypes
using DifferentiationInterfaceTest
using SparseConnectivityTracer
using SparseMatrixColorings

using DifferentiationInterfaceTest:
    default_scenarios,
    sparse_scenarios,
    complex_scenarios,
    complex_sparse_scenarios,
    static_scenarios,
    component_scenarios,
    gpu_scenarios

function MyAutoSparse(backend::AbstractADType)
    return AutoSparse(
        backend;
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )
end
