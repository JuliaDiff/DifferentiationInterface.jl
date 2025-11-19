using Pkg

@static if VERSION < v"1.11"
    DIT_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest")
    if isdir(DIT_PATH)
        Pkg.develop(; path = DIT_PATH)
    else
        Pkg.add("DifferentiationInterfaceTest")
    end
end

using ADTypes
using DifferentiationInterfaceTest
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

using DifferentiationInterfaceTest:
    default_scenarios,
    sparse_scenarios,
    complex_scenarios,
    complex_sparse_scenarios,
    static_scenarios,
    component_scenarios,
    gpu_scenarios,
    empty_scenarios

function MyAutoSparse(backend::AbstractADType)
    return AutoSparse(
        backend;
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(; postprocessing = true),
    )
end

safetypestab(symb) = VERSION < v"1.12-" ? symb : :none  # TODO: remove

LOGGING = get(ENV, "CI", "false") == "false"
