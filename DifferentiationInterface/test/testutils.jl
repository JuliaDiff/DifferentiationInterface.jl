using ADTypes
using SparseConnectivityTracer
using SparseMatrixColorings

function MyAutoSparse(backend::AbstractADType)
    return AutoSparse(
        backend;
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )
end
