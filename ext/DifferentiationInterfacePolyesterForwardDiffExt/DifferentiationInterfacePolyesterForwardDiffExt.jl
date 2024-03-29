module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes:
    AutoForwardDiff,
    AutoPolyesterForwardDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff
import DifferentiationInterface as DI
using DocStringExtensions
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using PolyesterForwardDiff.ForwardDiff: Chunk
using PolyesterForwardDiff.ForwardDiff.DiffResults: DiffResults

const AllAutoPolyForwardDiff{C} = Union{
    AutoPolyesterForwardDiff{C},AutoSparsePolyesterForwardDiff{C}
}

function single_threaded(::AutoPolyesterForwardDiff{C}) where {C}
    return AutoForwardDiff{C,Nothing}(nothing)
end

function single_threaded(::AutoSparsePolyesterForwardDiff{C}) where {C}
    return AutoSparseForwardDiff{C,Nothing}(nothing)
end

include("allocating.jl")
include("mutating.jl")

end # module
