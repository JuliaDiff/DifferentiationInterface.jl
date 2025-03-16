module DifferentiationInterfaceSparseMatrixColoringsExt

using ADTypes: ADTypes, AutoSparse, coloring_algorithm, dense_ad, sparsity_detector
import DifferentiationInterface as DI
using SparseMatrixColorings:
    AbstractColoringResult,
    ColoringProblem,
    coloring,
    column_colors,
    row_colors,
    ncolors,
    column_groups,
    row_groups,
    sparsity_pattern,
    decompress!
import SparseMatrixColorings as SMC

abstract type SparseJacobianPrep{SIG} <: DI.JacobianPrep{SIG} end

SMC.sparsity_pattern(prep::SparseJacobianPrep) = sparsity_pattern(prep.coloring_result)
SMC.column_colors(prep::SparseJacobianPrep) = column_colors(prep.coloring_result)
SMC.column_groups(prep::SparseJacobianPrep) = column_groups(prep.coloring_result)
SMC.row_colors(prep::SparseJacobianPrep) = row_colors(prep.coloring_result)
SMC.row_groups(prep::SparseJacobianPrep) = row_groups(prep.coloring_result)
SMC.ncolors(prep::SparseJacobianPrep) = ncolors(prep.coloring_result)

include("jacobian.jl")
include("jacobian_mixed.jl")
include("hessian.jl")

end
