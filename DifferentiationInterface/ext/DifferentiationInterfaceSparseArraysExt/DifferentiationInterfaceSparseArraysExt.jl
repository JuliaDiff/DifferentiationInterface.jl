module DifferentiationInterfaceSparseArraysExt

using ADTypes: ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using SparseArrays: SparseMatrixCSC, sparse, nonzeros, nnz

function DI.get_pattern(M::SparseMatrixCSC)
    S = similar(M, Bool)
    nonzeros(S) .= true
    return S
end

include("sparsity_detector.jl")

end
