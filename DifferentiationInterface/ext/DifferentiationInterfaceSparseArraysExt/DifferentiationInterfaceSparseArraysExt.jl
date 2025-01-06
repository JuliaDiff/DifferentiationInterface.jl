module DifferentiationInterfaceSparseArraysExt

using ADTypes: ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using SparseArrays: sparse

include("sparsity_detector.jl")

end
