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

## SMC overloads

abstract type SMCSparseJacobianPrep{SIG} <: DI.SparseJacobianPrep{SIG} end

SMC.sparsity_pattern(prep::DI.SparseJacobianPrep) = prep.sparsity
SMC.column_colors(prep::DI.SparseJacobianPrep) = column_colors(prep.coloring_result)
SMC.column_groups(prep::DI.SparseJacobianPrep) = column_groups(prep.coloring_result)
SMC.row_colors(prep::DI.SparseJacobianPrep) = row_colors(prep.coloring_result)
SMC.row_groups(prep::DI.SparseJacobianPrep) = row_groups(prep.coloring_result)
SMC.ncolors(prep::DI.SparseJacobianPrep) = ncolors(prep.coloring_result)

SMC.sparsity_pattern(prep::DI.SparseHessianPrep) = prep.sparsity
SMC.column_colors(prep::DI.SparseHessianPrep) = column_colors(prep.coloring_result)
SMC.column_groups(prep::DI.SparseHessianPrep) = column_groups(prep.coloring_result)
SMC.ncolors(prep::DI.SparseHessianPrep) = ncolors(prep.coloring_result)

include("jacobian.jl")
include("jacobian_mixed.jl")
include("hessian.jl")

end
